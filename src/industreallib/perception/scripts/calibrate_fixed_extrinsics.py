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

def _collect_tag_poses(config, detector, intrinsics, pipeline):
    """Gets the tag poses."""
    num_tag_detections = 0
    tag_ids, tag_poses_t, tag_poses_r = [], [], []

    tag_config = config.tag

    image = perception_utils.get_image(
        pipeline=pipeline, display_images=config.tag_detection.display_images
    )
    (
        are_tags_detected,
        _,
        tag_detection_results,
    ) = perception_utils.get_all_tag_poses_in_camera_frame(
        detector=detector,
        image=image,
        intrinsics=intrinsics,
        tag_length=tag_config.length,
        tag_active_pixel_ratio=tag_config.active_pixel_ratio,
    )

    if are_tags_detected:
        for tag_detection_result in tag_detection_results:
            (
                tag_id, 
                tag_pose_t,
                tag_pose_r,
                tag_corner_pixels,
                tag_family,
            ) = (
                tag_detection_result.id,
                tag_detection_result.pos,
                tag_detection_result.ori_mat,
                tag_detection_result.corner_pixels,
                tag_detection_result.family,
            )

            tag = tag_config[str(tag_id)]

            print(f"Tag {tag_id} detected.")
            num_tag_detections += 1

            tag_ids.append(tag_id)
            tag_poses_t.append(tag_pose_t.tolist())
            tag_poses_r.append(tag_pose_r.tolist())

            # Draw labels on image
            image_labeled = perception_utils.label_tag_detection(
                image=image, tag_corner_pixels=tag_corner_pixels, tag_family=tag_family
            )

            if num_tag_detections == config.tag_detection.num_detections:
                break

        if config.tag_detection.display_images:
            cv2.imshow("Tag Detection", image_labeled)
            cv2.waitKey(delay=2000)
            cv2.destroyAllWindows()

    else:
        print("Tags not detected.")

    return tag_ids, tag_poses_t, tag_poses_r

def _get_camera_pose(config, tag_ids, tag_poses_t_cam, tag_poses_r_cam):
    """
    Estimate the camera pose in world frame from observed tag poses in camera frame,
    and known tag poses in world frame from config.
    """
    tag_config = config.tag.tag_ids

    R_target2cam = []
    t_target2cam = []
    R_tag2world = []
    t_tag2world = []

    for tag_id, t_cam, R_cam in zip(tag_ids, tag_poses_t_cam, tag_poses_r_cam):
        if tag_id not in tag_config:
            print(f"Tag {tag_id} not found in config — skipping.")
            continue

        # Camera-frame observation (what the detector sees)
        R_target2cam.append(np.array(R_cam))           # Rotation: tag to camera
        t_target2cam.append(np.array(t_cam))           # Translation: tag to camera

        # World-frame ground truth (from config)
        tag_info = tag_config[str(tag_id)]
        R_world_to_tag = np.array(tag_info["pose_r"])  # Rotation: world to tag
        t_world_to_tag = np.array(tag_info["pose_t"])  # Translation: world to tag

        # Invert to get tag-to-world transform
        R_tag_to_world = R_world_to_tag.T
        t_tag_to_world = -R_tag_to_world @ t_world_to_tag

        R_tag2world.append(R_tag_to_world)
        t_tag2world.append(t_tag_to_world)

    if len(R_target2cam) < 3:
        raise ValueError("Not enough tag detections for camera-tag calibration.")

    # Run calibration to get camera-to-world transform
    camera_pos_r, camera_pos_t = cv2.calibrateHandEye(
        R_gripper2base=R_tag2world,
        t_gripper2base=t_tag2world,
        R_target2cam=R_target2cam,
        t_target2cam=t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    print("Camera extrinsics successfully computed.")
    print("Rotation:\n", camera_pos_r)
    print("Translation:\n", camera_pos_t.flatten())

    return camera_pos_r, camera_pos_t

def _save_extrinsics(file_name, camera_name, camera_pose_t, camera_pose_r):
    """Saves extrinsics for one camera to a multi-camera JSON file."""
    # Load existing file if it exists
    path = os.path.join(os.path.dirname(__file__), '..', 'io', file_name)
    if os.path.exists(path):
        with open(path, "r") as f:
            extrinsics = json.load(f)
    else:
        extrinsics = {}

    # Save under camera_name key
    extrinsics[camera_name] = {
        "position": camera_pose_t.tolist(),
        "orientation": camera_pose_r.tolist()
    }

    # Write back
    with open(path, "w") as f:
        json.dump(extrinsics, f, indent=2)

    print(f"Saved extrinsics for {camera_name}.")

if __name__ == "__main__":
    """Initializes the camera. Gets camera coordinates. Saves the extrinsics."""

    args = get_args()
    config = perception_utils.get_perception_config(
        file_name=args.perception_config_file_name, module_name="calibrate_fixed_extrinsics"
    )

    # Initialize AprilTag detector
    detector = apriltag.Detector(
        families=config.tag.type, quad_decimate=1.0, quad_sigma=0.0, decode_sharpening=0.25
    )
                   
    # Initialize cameras and get their world poses
    serials = perception_utils.get_connected_devices_serial()
    print(serials)
    
    for camera in config.camera.values():
        pipeline = perception_utils.get_camera_pipeline_with_serial(
            width=camera.image_width, height=camera.image_height, serial=camera.serial,
            use_depth=True, width_depth=camera.image_width_depth, height_depth=camera.image_height_depth
        )
        intrinsics = perception_utils.get_intrinsics(pipeline=pipeline)
        
        # Wait for pipeline to start
        for _ in range(30):  # ~1 second at 30 FPS
            pipeline.wait_for_frames()

        tag_ids, tag_poses_t_cam, tag_poses_r_cam = _collect_tag_poses(
            config=config,
            detector=detector,
            intrinsics=intrinsics,   
            pipeline=pipeline
        ) 

        camera_pose_r, camera_pose_t = _get_camera_pose(
            config=config,
            tag_ids=tag_ids,
            tag_poses_t_cam=tag_poses_t_cam,
            tag_poses_r_cam=tag_poses_r_cam,
        )

        _save_extrinsics(config=config, camera_name=camera.name, camera_pose_t=camera_pose_t, camera_pose_r=camera_pose_r)
