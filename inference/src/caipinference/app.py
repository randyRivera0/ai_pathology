from fastapi import FastAPI, Form, UploadFile

from caipinference.inference import InferenceService
from caipinference.models.grayscale_resnet50_toy_classifier import gray_classifier
from caipinference.models.rgb_resnet50_colon_classifier import rgb_classifier
from caipinference.schemas.prediction_result import PredictionResult

rgb_inference_service = InferenceService(rgb_classifier)
grayscale_inference_service = InferenceService(gray_classifier)

app = FastAPI()


@app.post("/predict/rgb", response_model=PredictionResult)
async def rgb_inference(file: UploadFile, tissue: str = Form(...)):
    image_bytes = await file.read()
    return rgb_inference_service.inference(image_bytes, tissue)


@app.post("/predict/gray", response_model=PredictionResult)
async def gray_inference(file: UploadFile, tissue: str = Form(...)):
    image_bytes = await file.read()
    return grayscale_inference_service.inference(image_bytes, tissue)
