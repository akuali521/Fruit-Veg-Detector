from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import requests
from PIL import Image
from io import BytesIO
import tensorflow as tf
import numpy as np
import os

app = FastAPI()

#load model
model = tf.keras.models.load_model("trained_model.h5")

#doc lable tu file txt
with open("labels.txt") as f:
    content = f.readlines()
labels = [label.strip() for label in content]

class ImageURL(BaseModel):
    image_url: str

def model_prediction(image, model):
    image = image.resize((64, 64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# uvicorn api:app --host 0.0.0.0 --port 8000


@app.post("/predict/")
async def predict(image_file: UploadFile = File(...)):
    try:
        contents = await image_file.read()
        image = Image.open(BytesIO(contents))
        result_index = model_prediction(image)
        
        # delete temp image
        os.remove(image_file.filename)
        
        return {"prediction": labels[result_index]}  
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error processing image")