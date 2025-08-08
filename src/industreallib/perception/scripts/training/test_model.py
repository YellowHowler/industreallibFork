import torch
from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Config 
IMG_PATH = "/home/turing/industreallibFork/collected_data/camera_1/rgb_0000.png"
CKPT_PATH = "/home/turing/industreallibFork/src/industreallib/perception/scripts/training/checkpoint.pt"
NUM_CLASSES = 3 

# Load model
model = maskrcnn_resnet50_fpn(num_classes=NUM_CLASSES)
model = torch.load(CKPT_PATH)  
model.eval()

# Load image
image = Image.open(IMG_PATH).convert("RGB")
transform = T.Compose([T.ToTensor()])
image_tensor = transform(image)

# Run model
with torch.no_grad():
    prediction = model([image_tensor])[0]

# Visualize
image_np = np.array(image)

# Show masks
for i in range(len(prediction["masks"])):
    mask = prediction["masks"][i, 0].numpy()
    score = prediction["scores"][i].item()
    label = prediction["labels"][i].item()

    if score > 0.5:
        plt.imshow(image_np)
        plt.imshow(mask, cmap="jet", alpha=0.5)
        plt.title(f"Label: {label}, Score: {score:.2f}")
        plt.axis("off")
        plt.show()
