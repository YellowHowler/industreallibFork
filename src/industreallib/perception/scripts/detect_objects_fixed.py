import os
import json
import argparse
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn
from PIL import Image

import pyrealsense2 as rs

# NVIDIA helpers
import industreallib.perception.scripts.perception_utils as perception_utils

def get_args():
    p = argparse.ArgumentParser()
    
    p.add_argument(
        "--perception_config_file_name", 
        "-p", 
        required=True,
        help="YAML config file (detect_objects section)"
    )
    
    return p.parse_args()

def main(perception_config_file_name: str):
    """
    Returns:
      object_coords: list of [x, y, theta] (meters, radians) in world frame
      object_labels: list of strings (label names)
    """
    # 1) load config
    cfg = perception_utils.get_perception_config(
        file_name=perception_config_file_name, module_name="detect_objects"
    )

    det_cfg   = cfg.object_detection
    scene_cfg = det_cfg.scene[det_cfg.scene.type]
    label_names = scene_cfg.label_names  # e.g., ['background','four_hole_base','four_hole_inserter']
    conf_thresh = float(det_cfg.confidence_thresh)

    # extrinsics file produced by calibrate_fixed_extrinsics
    ext_json = cfg.output.json_file_name if "json_file_name" in cfg.output else cfg.output.file_name
    ext_json_path = os.path.join(os.path.dirname(__file__), "..", "io", ext_json)

    # 2) set up both cameras
    cams = cfg.camera
    # Expect per-camera: name, serial, image_width, image_height, ckpt_path
    cam_entries = []
    for key in sorted(cams.keys()):
        cam = cams[key]
        entry = {
            "name": cam.name,
            "serial": cam.serial,
            "w": cam.image_width,
            "h": cam.image_height,
            "wd": cam.image_width_depth,
            "hd": cam.image_height_depth,
            "ckpt": cam.checkpoint_path,
        }
        if entry["ckpt"] is None:
            raise ValueError(f"Config must provide ckpt_path for camera '{cam.name}'.")
        cam_entries.append(entry)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models and pipelines
    for c in cam_entries:
        # RealSense pipeline
        c["pipeline"] = perception_utils.get_camera_pipeline_with_serial(
            width=c["w"], height=c["h"], serial=c["serial"],
            use_depth=True, width_depth=c["wd"], height_depth=c["hd"]
        )
        # warmup
        for _ in range(10):
            c["pipeline"].wait_for_frames()

        # intrinsics (for color stream)
        intr = perception_utils.get_intrinsics(pipeline=c["pipeline"])
        c["intr"] = build_intrinsics_dict(intr)

        # extrinsics world<-camera
        Rc, tc = load_extrinsics(ext_json_path, c["name"])
        c["Rcw"], c["tcw"] = Rc, tc

        # model
        # num_classes inferred from label_names length
        model = maskrcnn_resnet50_fpn(num_classes=len(label_names))
        model = torch.load(c["ckpt"], weights_only=False, map_location=device)
        model.eval().to(device)
        c["model"] = model

    # 3) grab frames, run inference, compute centers -> world
    all_dets = []
    for c in cam_entries:
        rgb, depth_m = align_realsense_frames(c["pipeline"])
        if rgb is None:
            continue

        pred = run_detector(c["model"], rgb)

        # per-instance extraction
        dets = []
        for i in range(len(pred["scores"])):
            score = float(pred["scores"][i].cpu().item())
            if score < conf_thresh:
                continue
            label_idx = int(pred["labels"][i].cpu().item())
            if label_idx <= 0 or label_idx >= len(label_names):
                continue

            mask_prob = pred["masks"][i, 0].detach().cpu().numpy()
            center_cam = center_from_mask_and_depth(
                mask_prob, depth_m, c["intr"], thresh=0.5, z_min=0.05, z_max=3.0, robust=True
            )
            if center_cam is None:
                continue

            # world pose
            p_w = to_world(center_cam, c["Rcw"], c["tcw"])
            theta_w = orientation_theta_world(mask_prob, depth_m, c["intr"], c["Rcw"], c["tcw"], thresh=0.5)

            dets.append({
                "label": label_names[label_idx],
                "xyz_world": p_w.tolist(),
                "theta": theta_w
            })

        all_dets.append(dets)

    # stop pipelines
    for c in cam_entries:
        c["pipeline"].stop()

    # 4) fuse detections across the two cameras (0.4 cam1 + 0.6 cam2)
    if len(all_dets) == 2:
        fused = fuse_two_cameras(all_dets[0], all_dets[1], w1=0.4, w2=0.6, match_radius=0.05)
    elif len(all_dets) == 1:
        fused = all_dets[0]
    else:
        fused = []

    # 5) pack outputs for the task interface
    object_coords = []  # list of [x, y, theta]
    object_labels = []  # list of strings
    for d in fused:
        x, y = d["xyz_world"][0], d["xyz_world"][1]
        theta = d["theta"]
        object_coords.append([float(x), float(y), float(theta)])
        object_labels.append(d["label"])

    return object_coords, object_labels

def load_extrinsics(json_path, camera_name):
    with open(json_path, "r") as f:
        ext = json.load(f)
    if camera_name not in ext:
        raise KeyError(f"Camera '{camera_name}' not found in {json_path}")
    t = np.array(ext[camera_name]["position"], dtype=np.float32).reshape(3, 1)
    R = np.array(ext[camera_name]["orientation"], dtype=np.float32)  # 3x3
    return R, t  # camera_in_world: p_w = R @ p_c + t


def center_from_mask_and_depth(mask_prob, depth_img_m, intr, thresh=0.5,
                               z_min=0.05, z_max=4.0, robust=True):
    """3D centroid in CAMERA frame from mask + full depth (meters)."""
    m = (mask_prob > thresh)
    if m.sum() < 20:
        return None

    ys, xs = np.where(m)
    Z = depth_img_m[ys, xs].astype(np.float32)
    valid = np.isfinite(Z) & (Z > z_min) & (Z < z_max)
    if valid.sum() < 20:
        return None

    xs = xs[valid].astype(np.float32)
    ys = ys[valid].astype(np.float32)
    Z  = Z[valid]

    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    X = (xs - cx) * Z / fx
    Y = (ys - cy) * Z / fy
    P = np.stack([X, Y, Z], axis=1)

    if robust and P.shape[0] >= 60:
        lo, hi = np.percentile(P, [5, 95], axis=0)
        keep = np.all((P >= lo) & (P <= hi), axis=1)
        if keep.any():
            P = P[keep]

    return P.mean(axis=0).astype(np.float32)


def orientation_theta_world(mask_prob, depth_img_m, intr, Rcw, tcw, thresh=0.5):
    """Estimate planar yaw (theta) in world XY via PCA on 3D points."""
    m = (mask_prob > thresh)
    ys, xs = np.where(m)
    if xs.size < 50:
        return 0.0

    Z = depth_img_m[ys, xs].astype(np.float32)
    valid = np.isfinite(Z) & (Z > 0.05) & (Z < 4.0)
    if valid.sum() < 50:
        return 0.0

    xs = xs[valid].astype(np.float32)
    ys = ys[valid].astype(np.float32)
    Z  = Z[valid]

    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    X = (xs - cx) * Z / fx
    Y = (ys - cy) * Z / fy
    Pw = (Rcw @ np.stack([X, Y, Z], axis=0) + tcw).T  # Nx3 world
    XY = Pw[:, :2] - Pw[:, :2].mean(0, keepdims=True)
    if XY.shape[0] < 2:
        return 0.0
    C = XY.T @ XY / (XY.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(C)
    v = eigvecs[:, np.argmax(eigvals)]  # principal axis on XY
    theta = np.arctan2(v[1], v[0])     # radians
    return float(theta)


def to_world(center_cam, Rcw, tcw):
    """camera -> world: p_w = R * p_c + t"""
    p = Rcw @ center_cam.reshape(3, 1) + tcw
    return p.flatten()


def align_realsense_frames(pipeline):
    """Return (rgb_uint8, depth_meters) aligned to color."""
    color, depth = perception_utils.get_image_color_depth(pipeline=pipeline, display_images=False)
    depth_scale = pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()
    depth_m = depth * depth_scale
    return color, depth_m


def build_intrinsics_dict(intr):
    return {
        "fx": intr.fx, "fy": intr.fy,
        "cx": intr.ppx, "cy": intr.ppy
    }


def run_detector(model, rgb):
    image = Image.fromarray(rgb).convert("RGB")
    tensor = T.ToTensor()(image).unsqueeze(0).to(next(model.parameters()).device)
    with torch.no_grad():
        pred = model(tensor)[0]
    return pred


def fuse_two_cameras(dets_cam1, dets_cam2, w1=0.4, w2=0.6, match_radius=0.05):
    """
    dets_cam*: list of dicts: {label, xyz_world (3,), theta}
    Weighted fuse if same label and within match_radius (meters).
    """
    fused = []
    used2 = set()

    for i, d1 in enumerate(dets_cam1):
        best_j, best_dist = None, 1e9
        for j, d2 in enumerate(dets_cam2):
            if j in used2 or d1["label"] != d2["label"]:
                continue
            dist = np.linalg.norm(np.array(d1["xyz_world"]) - np.array(d2["xyz_world"]))
            if dist < best_dist:
                best_dist, best_j = dist, j

        if best_j is not None and best_dist <= match_radius:
            p = w1 * np.array(d1["xyz_world"]) + w2 * np.array(dets_cam2[best_j]["xyz_world"])
            theta = float(np.arctan2(
                w1 * np.sin(d1["theta"]) + w2 * np.sin(dets_cam2[best_j]["theta"]),
                w1 * np.cos(d1["theta"]) + w2 * np.cos(dets_cam2[best_j]["theta"])
            ))
            fused.append({"label": d1["label"], "xyz_world": p.tolist(), "theta": theta})
            used2.add(best_j)
        else:
            fused.append(d1)

    # add leftover from cam2
    for j, d2 in enumerate(dets_cam2):
        if j not in used2:
            fused.append(d2)

    return fused

if __name__ == "__main__":
    args = get_args()
    coords, labels = main(perception_config_file_name=args.perception_config_file_name)
    print("Object labels:", labels)
    print("Object coords (x, y, theta):", coords)