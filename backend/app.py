
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import numpy as np
import tensorflow as tf
import json


# Load model
model = tf.keras.models.load_model("/content/drive/MyDrive/gesture_navigator_project/models/gesture_model.keras")

# Load class names
with open("/content/drive/MyDrive/gesture_navigator_project/models/class_names.json", "r") as f:
    class_names = json.load(f)

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Gesture Navigator API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    image = image.resize((224, 224))

    image = np.array(image, dtype=np.float32)
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image, verbose=0)

    class_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    return {
        "gesture": class_names[class_index],
        "confidence": confidence
    }
