import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.transforms.functional import pil_to_tensor
from torchvision.utils import draw_bounding_boxes

weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
categories = weights.meta["categories"]
img_preprocess = weights.transforms()

@st.cache_resource
def load_model():
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5)
    model.eval()
    return model

model = load_model()

def make_prediction(img):
    img_processed = img_preprocess(img)
    prediction = model(img_processed.unsqueeze(0))[0]
    prediction["labels"] = [categories[label] for label in prediction["labels"]]
    return prediction

def create_image_with_bboxes(img, prediction):
    img_with_bboxes = draw_bounding_boxes(img, boxes=prediction["boxes"], labels=prediction["labels"],
                                          colors=["red" if label=="person" else "green" for label in prediction["labels"]], width=2)
    img_with_bboxes_np = img_with_bboxes.numpy().transpose(1, 2, 0)
    return img_with_bboxes_np

def process_image(image):
    # Convert image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)

st.title("Object Detector :tea: :coffee:")
upload = st.file_uploader(label="Upload Image Here:", type=["png", "jpg", "jpeg"])

if upload:
    image = np.array(bytearray(upload.read()), dtype=np.uint8)
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Process the image
    processed_img = process_image(img)

    prediction = make_prediction(processed_img)
    img_with_bbox = create_image_with_bboxes(pil_to_tensor(processed_img), prediction)

    # Count the number of objects
    num_objects = len(prediction["labels"])

    # Displaying original image and image with bounding boxes side by side
    col1, col2 = st.columns(2)
    with col1:
        st.header("Original Image")
        st.image(processed_img, use_column_width=True)

    with col2:
        st.header("Image with Bounding Boxes")
        st.image(img_with_bbox, use_column_width=True)

    st.header("Predicted Probabilities")
    st.write(prediction)

    st.header("Number of Objects Detected")
    st.write(num_objects)
