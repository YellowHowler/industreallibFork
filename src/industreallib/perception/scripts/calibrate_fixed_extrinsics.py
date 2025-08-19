# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].
"""IndustRealLib: Calibrate extrinsics script.

This script is a standalone script that calibrates the extrinsics of a wrist-
mounted Intel RealSense camera on a Franka robot. The script loads parameters
for the calibration procedure from a specified YAML file, runs a calibration
procedure where multiple robot and tag poses are recorded, computes the
extrinsics, and writes them to a JSON file.

Typical usage example:
python calibrate_extrinsics.py -p perception.yaml
"""

# Standard Library
import argparse
import json
import os
import time

# Third Party
import cv2
import numpy as np
import pupil_apriltags as apriltag 

# NVIDIA
import industreallib.perception.scripts.perception_utils as perception_utils

def get_args():
    """Gets arguments from the command line."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-p",
        "--perception_config_file_name",
        required=True,
        help="Perception configuration to load",
    )

    args = parser.parse_args()

    return args


def to_T(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T

def invert_T(R, t):
    Rinv = R.T
    tinv = -Rinv @ t
    return Rinv, tinv

def avg_rotation(Rs):
    """Chordal mean via SVD."""
    M = np.zeros((3, 3), dtype=np.float64)
    for R in Rs:
        M += R
    U, _, Vt = np.linalg.svd(M)
    Rm = U @ Vt
    if np.linalg.det(Rm) < 0:
        U[:, -1] *= -1
        Rm = U @ Vt
    return Rm

def consistency_score(R_list, t_list):
    """Lower is better: rotation dispersion + translation dispersion."""
    Rm = avg_rotation(R_list)
    tm = np.mean(np.stack(t_list, axis=0), axis=0)
    rot_err = sum(np.linalg.norm(R - Rm, ord='fro') for R in R_list)
    trans_err = sum(np.linalg.norm(t - tm) for t in t_list)
    return rot_err + trans_err, Rm, tm

def print_pose_quickcheck(R_wc, t_wc, label="camera"):
    fwd = R_wc[:, 2]  # camera +Z in world
    fwd = fwd / np.linalg.norm(fwd)
    ang_to_plusY = np.degrees(np.arccos(np.clip(fwd[1], -1, 1)))         # 0° if pointing +Y
    ang_to_minusZ = np.degrees(np.arccos(np.clip(-fwd[2], -1, 1)))       # 0° if pointing -Z (down)
    print(f"[{label}] pos (m): {t_wc.round(4)}")
    print(f"[{label}] forward (world): {fwd.round(4)}")
    print(f"[{label}] angle to +Y: {ang_to_plusY:.1f}°, angle to -Z (down): {ang_to_minusZ:.1f}°")

# --------------------------- Detection collection ---------------------------

def _collect_tag_poses(config, detector, intrinsics, pipeline, camera_name="camera"):
    """Acquire one RGB frame, detect all tags, store per-tag pose (R,t) and save annotated image."""
    tag_cfg = config.tag
    num_needed = config["tag_detection"]["num_detection"]

    image = perception_utils.get_image(
        pipeline=pipeline, display_images=False  # stay headless-safe
    )

    are_tags_detected, _, tag_detection_results = perception_utils.get_all_tag_poses_in_camera_frame(
        detector=detector,
        image=image,
        intrinsics=intrinsics,
        tag_length=tag_cfg.length,
        tag_active_pixel_ratio=tag_cfg.active_pixel_ratio,
    )
    
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(os.path.join("outputs", f"{camera_name}_image.png"), image)

    tag_ids, t_list, R_list = [], [], []
    if are_tags_detected:
        for det in tag_detection_results:
            tid = det["id"]
            if str(tid) not in config.tag["tag_ids"]:
                continue

            tag_ids.append(tid)
            t_list.append(det["pos"])        # assumed: TAG in CAMERA (t_tag_cam)
            R_list.append(det["ori_mat"])    # assumed: TAG in CAMERA (R_tag_cam)

            # draw label
            image = perception_utils.label_tag_detection(
                image=image,
                tag_corner_pixels=det["corner_pixels"],
                tag_family=det["family"],
                tag_id=tid,
            )
            if len(tag_ids) >= num_needed:
                break

        os.makedirs("outputs", exist_ok=True)
        cv2.imwrite(os.path.join("outputs", f"{camera_name}_detected_tags.png"), image)

        if len(tag_ids) == 0:
            print("Tags detected, but none matched config IDs.")
    else:
        print("No tags detected.")

    return tag_ids, t_list, R_list

# --------------------------- Core extrinsics solve ---------------------------

def _get_camera_pose(config, tag_ids, tag_poses_t_cam, tag_poses_r_cam, camera_label="camera"):
    """
    Estimate ^W T_C from:
      - Config (per-tag): pose_r, pose_t (either TAG->WORLD or WORLD->TAG)
      - Detections (per-tag): (R, t) (either TAG-in-CAMERA or CAMERA-in-TAG)
    Tries both conventions and returns the best by internal consistency.
    """
    tag_cfg = config.tag["tag_ids"]

    # Pack all tag data
    tags = []
    for tid, t_cam, R_cam in zip(tag_ids, tag_poses_t_cam, tag_poses_r_cam):
        if str(tid) not in tag_cfg:
            continue
        t_cam = np.asarray(t_cam, dtype=np.float64).reshape(3)
        R_cam = np.asarray(R_cam, dtype=np.float64).reshape(3, 3)
        R_cfg = np.asarray(tag_cfg[str(tid)]["pose_r"], dtype=np.float64).reshape(3, 3)
        t_cfg = np.asarray(tag_cfg[str(tid)]["pose_t"], dtype=np.float64).reshape(3)
        tags.append(dict(id=tid, R_cfg=R_cfg, t_cfg=t_cfg, R_obs=R_cam, t_obs=t_cam))

    if len(tags) == 0:
        raise ValueError("No valid tag observations matched the config.")

    candidates = []
    # cfg_convention: True = TAG->WORLD in YAML, False = WORLD->TAG
    # obs_convention: True = TAG-in-CAMERA from detector, False = CAMERA-in-TAG
    for cfg_is_tag_to_world in (True, False):
        for obs_is_tag_in_cam in (True, False):
            R_wc_list, t_wc_list = [], []
            for tag in tags:
                # Config convention
                if cfg_is_tag_to_world:
                    R_tw, t_tw = tag["R_cfg"], tag["t_cfg"]
                else:
                    # WORLD->TAG → invert once to get TAG->WORLD
                    R_tw = tag["R_cfg"].T
                    t_tw = -R_tw @ tag["t_cfg"]

                # Observation convention
                if obs_is_tag_in_cam:
                    R_ct, t_ct = tag["R_obs"], tag["t_obs"]    # TAG-in-CAMERA
                else:
                    # CAMERA-in-TAG → invert to get TAG-in-CAMERA
                    R_ct, t_ct = invert_T(tag["R_obs"], tag["t_obs"])

                TwT = to_T(R_tw, t_tw)
                TcT = to_T(R_ct, t_ct)
                TwC = TwT @ np.linalg.inv(TcT)

                R_wc_list.append(TwC[:3, :3])
                t_wc_list.append(TwC[:3, 3])

            score, Rm, tm = consistency_score(R_wc_list, t_wc_list)
            candidates.append(dict(score=score,
                                   R=Rm, t=tm,
                                   cfg_is_tag_to_world=cfg_is_tag_to_world,
                                   obs_is_tag_in_cam=obs_is_tag_in_cam))

    # Pick best (lowest dispersion)
    candidates.sort(key=lambda d: d["score"])
    best = candidates[0]

    print(f"[{camera_label}] picked conventions:"
          f" cfg_is_TAG->WORLD={best['cfg_is_tag_to_world']},"
          f" obs_is_TAG_in_CAMERA={best['obs_is_tag_in_cam']},"
          f" score={best['score']:.4f}")

    # Sanity print
    print_pose_quickcheck(best["R"], best["t"], label=camera_label)

    return best["R"], best["t"].reshape(3, 1)

# --------------------------- Save ---------------------------

def _save_extrinsics(file_name, camera_name, camera_pose_t, camera_pose_r):
    """Saves extrinsics for one camera to a multi-camera JSON file."""
    path = os.path.join(os.path.dirname(__file__), '..', 'io', file_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        with open(path, "r") as f:
            extrinsics = json.load(f)
    else:
        extrinsics = {}

    extrinsics[camera_name] = {
        "position": camera_pose_t.tolist(),
        "orientation": camera_pose_r.tolist()
    }

    with open(path, "w") as f:
        json.dump(extrinsics, f, indent=2)

    print(f"Saved extrinsics for {camera_name} → {path}")

# --------------------------- Main ---------------------------

if __name__ == "__main__":
    """Initialize cameras, estimate ^W T_C, and save extrinsics."""
    args = get_args()
    config = perception_utils.get_perception_config(
        file_name=args.perception_config_file_name,
        module_name="calibrate_fixed_extrinsics"
    )

    detector = apriltag.Detector(
        families=config.tag.type,
        quad_decimate=1.0,
        quad_sigma=0.0,
        decode_sharpening=0.25
    )

    serials = perception_utils.get_connected_devices_serial()
    print("Connected RealSense serials:", serials)

    for camera in config.camera.values():
        pipeline = perception_utils.get_camera_pipeline_with_serial(
            width=camera.image_width, height=camera.image_height, serial=camera.serial,
            use_depth=True, width_depth=camera.image_width_depth, height_depth=camera.image_height_depth
        )
        intrinsics = perception_utils.get_intrinsics(pipeline=pipeline)

        # Warm-up frames
        for _ in range(30):
            pipeline.wait_for_frames()

        tag_ids, tag_t_cam, tag_R_cam = _collect_tag_poses(
            config=config,
            detector=detector,
            intrinsics=intrinsics,
            pipeline=pipeline,
            camera_name=camera.name,
        )

        if len(tag_ids) < 1:
            print(f"[{camera.name}] No tags. Skipping.")
            pipeline.stop()
            continue

        R_wc, t_wc = _get_camera_pose(
            config=config,
            tag_ids=tag_ids,
            tag_poses_t_cam=tag_t_cam,
            tag_poses_r_cam=tag_R_cam,
            camera_label=camera.name
        )

        _save_extrinsics(
            file_name=config.output["file_name"],
            camera_name=camera.name,
            camera_pose_t=t_wc,
            camera_pose_r=R_wc
        )

        pipeline.stop()

    cv2.destroyAllWindows() 