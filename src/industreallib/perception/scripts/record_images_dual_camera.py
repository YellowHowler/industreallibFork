import argparse
import json
import os
import cv2
import numpy as np
import pyrealsense2 as rs
import industreallib.perception.scripts.perception_utils as perception_utils

SAVE_DIR = "./collected_data"
os.makedirs(SAVE_DIR, exist_ok=True)

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

def main():
    args = get_args()
    config = perception_utils.get_perception_config(
        file_name=args.perception_config_file_name, module_name="calibrate_fixed_extrinsics"
    )

    # Initialize cameras and get their world poses
    serials = perception_utils.get_connected_devices_serial()
    print(serials)

    pipelines = {}
    for _, cam in config.camera.items():
        pipeline = perception_utils.get_camera_pipeline_with_serial(
            width=cam["image_width"], height=cam["image_height"], serial=cam["serial"], 
            use_depth=True, width_depth=cam["image_width_depth"], height_depth=cam["image_height_depth"]
        )
        pipelines[cam["name"]] = pipeline

        # Create per-camera folder
        cam_folder = os.path.join(SAVE_DIR, cam["name"])
        os.makedirs(cam_folder, exist_ok=True)
    
    print("Press SPACE to capture, ESC to quit.")
    
    counter = 0
    while True:
        print(f"Capturing frame {counter}...")

        for _, cam in config.camera.items():
            name = cam["name"]
            pipeline = pipelines[name]
            color_image, depth_image = perception_utils.get_image_color_depth(pipeline, display_images=False)

            cam_folder = os.path.join(SAVE_DIR, cam["name"])

            # Save RGB
            rgb_path = os.path.join(cam_folder, f"rgb_{counter:04d}.png")
            cv2.imwrite(rgb_path, color_image)

            # Save Depth (16-bit PNG)
            depth_path = os.path.join(cam_folder, f"depth_{counter:04d}.png")
            cv2.imwrite(depth_path, depth_image)

            print(f"Saved to {rgb_path} and {depth_path}")

        counter += 1

        input("Press ENTER to capture next frame or CTRL+C to stop...")

    # Cleanup
    for pipeline in pipelines.values():
        pipeline.stop()
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    main()
