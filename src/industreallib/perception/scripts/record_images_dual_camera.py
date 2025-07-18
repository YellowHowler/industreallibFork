import os
import cv2
import numpy as np
import pyrealsense2 as rs
import industreallib.perception.scripts.perception_utils as perception_utils

SAVE_DIR = "./collected_data"
os.makedirs(SAVE_DIR, exist_ok=True)

# Camera config 
CAMERAS = [
    {"name": "camera_1", "serial": "f1380463"},
    {"name": "camera_2", "serial": "f1421239"},
]

def capture_rgb_depth(pipeline):
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    if not color_frame or not depth_frame:
        raise RuntimeError("Failed to get frames.")
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    return color_image, depth_image

def main():
    pipelines = {}
    for cam in CAMERAS:
        pipeline = perception_utils.get_camera_pipeline_with_serial(
            width=640, height=480, serial=cam["serial"]
        )
        pipelines[cam["name"]] = pipeline
    
    print("Press SPACE to capture, ESC to quit.")
    
    counter = 0
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32:  # Spacebar
            print(f"Capturing frame {counter}...")

            for cam in CAMERAS:
                name = cam["name"]
                pipeline = pipelines[name]
                color, depth = capture_rgb_depth(pipeline)

                # Create per-camera folder
                cam_folder = os.path.join(SAVE_DIR, name)
                os.makedirs(cam_folder, exist_ok=True)

                # Save RGB
                rgb_path = os.path.join(cam_folder, f"rgb_{counter:04d}.png")
                cv2.imwrite(rgb_path, color)

                # Save Depth (16-bit PNG)
                depth_path = os.path.join(cam_folder, f"depth_{counter:04d}.png")
                cv2.imwrite(depth_path, depth)

                print(f"Saved to {rgb_path} and {depth_path}")

            counter += 1

    # Cleanup
    for pipeline in pipelines.values():
        pipeline.stop()
    cv2.destroyAllWindows()
    print("✅ Done.")

if __name__ == "__main__":
    main()
