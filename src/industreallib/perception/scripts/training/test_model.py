import torch
from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn
import torchvision.transforms as T
from PIL import Image
from PIL import ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
import os

# Config 
IMG_PATH = "/home/fenchel/industreallibFork/perception_training/camera2/train/camera2_rgb_0009.png"
CKPT_PATH = "/home/fenchel/industreallibFork/src/industreallib/perception/scripts/training/checkpoint.pt"
NUM_CLASSES = 3 

# Load model
model = maskrcnn_resnet50_fpn(num_classes=NUM_CLASSES)
model = torch.load(CKPT_PATH, weights_only=False)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Load image
image = Image.open(IMG_PATH).convert("RGB")
transform = T.Compose([T.ToTensor()])
image_tensor = transform(image)

# Run model
with torch.no_grad():
    prediction = model([image_tensor.to(device)])[0]

# Ensure image is RGB
image_rgb = image.convert("RGB")
image_np = np.array(image_rgb).astype(np.uint8)

# Convert to float32 for blending
blended = image_np.astype(np.float32) / 255.0

# Set up drawing
output_img = Image.fromarray((blended * 255).astype(np.uint8))
draw = ImageDraw.Draw(output_img)

# Load font
try:
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
except:
    font = ImageFont.load_default()

# Overlay masks with random colors
for i in range(len(prediction["masks"])):
    score = prediction["scores"][i].item()
    label = prediction["labels"][i].item()

    if score > 0.5:
        mask = prediction["masks"][i, 0].detach().cpu().numpy()
        mask_bin = (mask > 0.5).astype(np.uint8)

        # Generate a random color
        color = np.random.rand(3)

        alpha = 0.8
        
        # Apply color to mask
        for c in range(3):
            blended[:, :, c] = np.where(
                mask_bin == 1,
                blended[:, :, c] * (1 - alpha) + color[c] * alpha,
                blended[:, :, c]
            )

        # Draw label text
        box = prediction["boxes"][i].detach().cpu().numpy()
        x1, y1 = int(box[0]), int(box[1])
        draw.text((x1, y1 - 10), f"Label {label} ({score:.2f})", fill="white", font=font)

# Save image
blended_img = Image.fromarray((blended * 255).astype(np.uint8))
blended_img.save("/home/fenchel/industreallibFork/outputs/mask_overlay_fixed.png")
print("✅ Saved overlay with masks and labels!")