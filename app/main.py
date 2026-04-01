from fastapi import FastAPI , UploadFile , File , Form
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
import shutil

sys.path.append('.')
from src.prediction import predict 
from src.prediction import reload_model
from src.model import retrain

app=FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]

)
@app.get("/health")
async def health():
    return {"status": "running"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    temp_path = "temp_image.jpg"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    result = predict(temp_path)
    os.remove(temp_path)
    return result

@app.post("/upload")
async def upload_images(
    files: list[UploadFile] = File(...),
    class_name: str = Form(...)
):
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_folder = os.path.join(BASE_DIR, "data", "train", class_name)
    os.makedirs(save_folder, exist_ok=True)
    for file in files:
        filename = file.filename or "unknown.jpg"
        file_path = os.path.join(save_folder, filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    return {"message": f"{len(files)} files uploaded to {class_name}"}

@app.post("/retrain")
async def retrain_model():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(BASE_DIR, "data", "train")
    retrain(data_dir)
    reload_model()
    return {"message": "Retraining complete"}