import torch
import torchvision
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
import gradio as gr
import random

# Load pre-trained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# COCO dataset labels
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Assign random colors for each label
random.seed(42)  # Fix seed for consistent colors every run
label_colors = {label: (random.randint(0,255), random.randint(0,255), random.randint(0,255)) for label in COCO_INSTANCE_CATEGORY_NAMES}

def detect_objects(image, threshold):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    img = transform(image)
    with torch.no_grad():
        prediction = model([img])

    boxes = prediction[0]['boxes']
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']

    draw = ImageDraw.Draw(image)
    for box, label, score in zip(boxes, labels, scores):
        if score > threshold:
            box = [int(i) for i in box.tolist()]
            label_name = COCO_INSTANCE_CATEGORY_NAMES[label.item()]
            color = label_colors.get(label_name, (255, 0, 0))  # fallback to red
            draw.rectangle(box, outline=color, width=3)
            draw.text((box[0]+5, box[1]+5), f"{label_name} ({score:.2f})", fill=color)
    
    return image

# Gradio App
demo = gr.Interface(
    fn=detect_objects,
    inputs=[
        gr.Image(type="pil"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.6, label="Confidence Threshold")
    ],
    outputs=gr.Image(type="pil"),
    title="AI Object Detection App",
    description="Upload an image and detect objects with bounding boxes using a pre-trained Faster R-CNN model! Adjust the confidence threshold with the slider.",
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch(share=True)
