import cv2
import numpy as np
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import os

# Load pre-trained model
model = MobileNetV2(weights='imagenet')

# Load image and resize it to 224x224 pixels
img_path = 'images/lion3.jpg'

if not os.path.isfile(img_path):
    raise ValueError(f"Image file {img_path} does not exist")

try:
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to load image file {img_path}")
    img = cv2.resize(img, (224, 224))
except cv2.error as e:
    raise ValueError(f"Error while processing image file {img_path}: {e}")

# Preprocess input image
img = np.expand_dims(img, axis=0)
img = preprocess_input(img)

# Predict animal type and output probabilities
try:
    preds = model.predict(img)
    preds_decoded = decode_predictions(preds, top=5)[0]
    print("Predicted animal types and probabilities:")
    for pred in preds_decoded:
        print(f"{pred[1]}: {pred[2]*100:.2f}%")
except Exception as e:
    print("Failed to predict animal type:", e)
