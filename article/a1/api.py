from io import BytesIO

import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image

app = FastAPI()

model = torch.jit.load("model.jit.pt").eval()

IMAGE_SIZE = 128, 128


def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file)).convert("RGB")
    if IMAGE_SIZE:
        image = image.resize(IMAGE_SIZE)
    return image


def preprocess(image):
    img_arr = np.asarray(image)
    img_arr = torch.as_tensor(img_arr).permute(2, 0, 1)
    img_arr = img_arr.unsqueeze(0)
    img_arr = img_arr / 255.0
    return img_arr


@torch.inference_mode()
def predict(image):
    img_arr = preprocess(image)
    output = model(img_arr)

    output = output.softmax(1).argmax()
    return output.item()


@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    prediction = predict(image)

    return prediction
