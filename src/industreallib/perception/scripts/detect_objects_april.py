# Standard Library
import argparse
import json
import os

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

def main(perception_config_file_name):
    """Detect AprilTags from multiple cameras and return world-frame coordinates + labels."""
    config = perception_utils.get_perception_config(
        file_name=perception_config_file_name, module_name="detect_objects_april"
    )

    # Load camera extrinsics file
    extrinsics_path = os.path.join(os.path.dirname(__file__), '..', 'io', config.output.file_name)
    with open(extrinsics_path, "r") as f:
        camera_extrinsics = json.load(f)

    # Prepare tag detector
    detector = apriltag.Detector(
        families=config.tag.type, quad_decimate=1.0, quad_sigma=0.0, decode_sharpening=0.25
    )

    # Accumulate detections across cameras per tag
    tag_detections = {tag_id: [] for tag_id in config.object_detection.tag.tag_ids.keys()}

    # Loop over all cameras
    for _, cam_cfg in config.camera.items():
        cam_name = cam_cfg.name
        if cam_name not in camera_extrinsics:
            print(f"Camera extrinsics for '{cam_name}' not found. Skipping.")
            continue

        # Build T_cam_in_world from extrinsics
        cam_pose = camera_extrinsics[cam_name]
        T_cam_in_world = np.eye(4)
        T_cam_in_world[:3, :3] = np.array(cam_pose["orientation"])
        T_cam_in_world[:3, 3] = np.array(cam_pose["position"])

        # Start camera and grab image
        pipeline = perception_utils.get_camera_pipeline_with_serial(
            width=cam_cfg.image_width, height=cam_cfg.image_height, serial=cam_cfg.serial
        )
        image = perception_utils.get_image(
            pipeline=pipeline,
            display_images=config.object_detection.display_images
        )
        intrinsics = perception_utils.get_intrinsics(pipeline)

        # Detect each tag
        image = perception_utils.get_image(
            pipeline=pipeline, display_images=config.tag_detection.display_images
        )
        (
            are_tags_detected,
            num_detections,
            tag_detection_results,
        ) = perception_utils.get_all_tag_poses_in_camera_frame(
            detector=detector,
            image=image,
            intrinsics=intrinsics,
            tag_length=tag.length,
            tag_active_pixel_ratio=tag.active_pixel_ratio,
        )

        if are_tags_detected:
            for tag_detection_result in tag_detection_results:
                (
                    tag_id, 
                    tag_pose_t,
                    tag_pose_r,
                ) = (
                    tag_detection_result["id"],
                    tag_detection_result["pos"],
                    tag_detection_result["ori_mat"],
                )

                tag = config.tag.tag_ids[str(tag_id)]

                # Tag pose in camera frame → world frame
                T_tag_in_cam = np.eye(4)
                T_tag_in_cam[:3, :3] = np.array(tag_pose_r)
                T_tag_in_cam[:3, 3] = np.array(tag_pose_t).flatten()
                T_tag_in_world = T_cam_in_world @ T_tag_in_cam

                # Extract [x, y, theta]
                offset_tag_to_object = np.eye(4)
                offset_tag_to_object[:3, 3] = tag.offset if "offset" in tag else [0.0, 0.0, 0.0]

                T_object_in_world = T_tag_in_world @ offset_tag_to_object
                x, y = T_object_in_world[0, 3], T_object_in_world[1, 3]
                theta = np.arctan2(T_object_in_world[1, 0], T_object_in_world[0, 0])
                
                tag_detections[tag_id].append([x, y, theta])

                print(f"Tag {tag_id} seen by {cam_name}: (x={x:.3f}, y={y:.3f}, θ={theta:.2f})")
        
        else:
            print("Tags not detected.")

    # Average detections
    box_real_coords = []
    labels_text = []
    for tag_id, poses in tag_detections.items():
        if not poses:
            print(f"Tag {tag_id} not detected by any camera.")
            continue
        poses = np.array(poses)
        x_mean, y_mean = poses[:, 0].mean(), poses[:, 1].mean()
        # Average theta using atan2 of sin and cos
        sin_sum = np.sin(poses[:, 2]).sum()
        cos_sum = np.cos(poses[:, 2]).sum()
        theta_mean = np.arctan2(sin_sum, cos_sum)

        label = config.tag.tag_ids[str(tag_id)].label
        box_real_coords.append([x_mean, y_mean, theta_mean])
        labels_text.append(label)

        print(f"Final for {label}: (x={x_mean:.3f}, y={y_mean:.3f}, θ={theta_mean:.2f}) from {len(poses)} views")

    return box_real_coords, labels_text

if __name__ == "__main__":
    """Gets arguments. Runs the script."""
    args = get_args()

    main(perception_config_file_name=args.perception_config_file_name)
