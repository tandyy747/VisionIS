import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import joblib
import pickle
import os
import io

# ---------------------- Load các mô hình ----------------------

# VGG16
vgg16_model_nofinetune = load_model("models/vgg16nofintune/best_model_vgg16.h5")
vgg16_model_finetune = load_model("models/vgg16finetune/best_model_vgg16_finetune.h5")
svm_vgg16_nofinetune = joblib.load("models/vgg16svmnofinetune/svm_model.pkl")
feature_vgg16_nofinetune = load_model("models/vgg16svmnofinetune/feature_extractor.h5")
svm_vgg16_finetune = joblib.load("models/vgg16svmfinetune/svm_model.pkl")
feature_vgg16_finetune = load_model("models/vgg16svmfinetune/feature_extractor.h5")
with open("models/vgg16svmnofinetune/class_names.pkl", "rb") as f:
    class_names_vgg16 = pickle.load(f)

# ResNet50
resnet_model_nofinetune = load_model("models/resnet50nofinetune/best_model_resnet50.h5")
resnet_model_finetune = load_model("models/resnet50finetune/best_model_resnet50_finetune.h5")
svm_resnet_nofinetune = joblib.load("models/resnet50svmnofinetune/svm_model.pkl")
feature_resnet_nofinetune = load_model("models/resnet50svmnofinetune/feature_extractor.h5")
svm_resnet_finetune = joblib.load("models/resnet50svmfinetune/svm_model.pkl")
feature_resnet_finetune = load_model("models/resnet50svmfinetune/feature_extractor.h5")
with open("models/resnet50svmnofinetune/class_names.pkl", "rb") as f:
    class_names_resnet = pickle.load(f)

# MobileNetV2
mobilenet_model_nofinetune = load_model("models/mobilenetv2nofinetune/best_model_mobilenetv2.h5")
mobilenet_model_finetune = load_model("models/mobilenetv2finetune/best_model_mobilenetv2_finetune.h5")
svm_mobilenet_nofinetune = joblib.load("models/mobilenetv2svmnofintune/svm_model.pkl")
feature_mobilenet_nofinetune = load_model("models/mobilenetv2svmnofintune/feature_extractor.h5")
svm_mobilenet_finetune = joblib.load("models/mobilenetv2svmfinetune/svm_model.pkl")
feature_mobilenet_finetune = load_model("models/mobilenetv2svmfinetune/feature_extractor.h5")
with open("models/mobilenetv2svmnofintune/class_names.pkl", "rb") as f:
    class_names_mobilenet = pickle.load(f)

# ---------------------- Hàm xử lý ảnh ----------------------

def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    img = img.resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# ---------------------- Hàm dự đoán ----------------------

def predict_end2end(img, model, class_names):
    processed = preprocess_image(img)
    prediction = model.predict(processed)
    return class_names[np.argmax(prediction)]

def predict_with_svm(img, feature_extractor, svm_model, class_names):
    processed = preprocess_image(img)
    features = feature_extractor.predict(processed)
    return class_names[svm_model.predict(features)[0]]

# ---------------------- FastAPI ----------------------

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_HANDLERS = {
    "VGG16 không finetune": lambda img: predict_end2end(img, vgg16_model_nofinetune, class_names_vgg16),
    "VGG16 có finetune": lambda img: predict_end2end(img, vgg16_model_finetune, class_names_vgg16),
    "VGG16 + SVM không finetune": lambda img: predict_with_svm(img, feature_vgg16_nofinetune, svm_vgg16_nofinetune, class_names_vgg16),
    "VGG16 + SVM có finetune": lambda img: predict_with_svm(img, feature_vgg16_finetune, svm_vgg16_finetune, class_names_vgg16),

    "ResNet50 không finetune": lambda img: predict_end2end(img, resnet_model_nofinetune, class_names_resnet),
    "ResNet50 có finetune": lambda img: predict_end2end(img, resnet_model_finetune, class_names_resnet),
    "ResNet50 + SVM không finetune": lambda img: predict_with_svm(img, feature_resnet_nofinetune, svm_resnet_nofinetune, class_names_resnet),
    "ResNet50 + SVM có finetune": lambda img: predict_with_svm(img, feature_resnet_finetune, svm_resnet_finetune, class_names_resnet),

    "MobileNetV2 không finetune": lambda img: predict_end2end(img, mobilenet_model_nofinetune, class_names_mobilenet),
    "MobileNetV2 có finetune": lambda img: predict_end2end(img, mobilenet_model_finetune, class_names_mobilenet),
    "MobileNetV2 + SVM không finetune": lambda img: predict_with_svm(img, feature_mobilenet_nofinetune, svm_mobilenet_nofinetune, class_names_mobilenet),
    "MobileNetV2 + SVM có finetune": lambda img: predict_with_svm(img, feature_mobilenet_finetune, svm_mobilenet_finetune, class_names_mobilenet),
}

@app.post("/predict/")
async def predict(
    images: List[UploadFile] = File(...),
    models: List[str] = Form(...)
):
    results = []

    for image_file in images:
        content = await image_file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")

        result_per_image = {
            "imageName": image_file.filename,
            "imageUrl": f"/static/{image_file.filename}",
            "classifications": []
        }

        for model in models:
            try:
                label = MODEL_HANDLERS[model](img)
            except KeyError:
                label = "❌ Mô hình chưa được hỗ trợ"
            except Exception as e:
                label = f"Lỗi: {str(e)}"

            result_per_image["classifications"].append({
                "modelName": model,
                "result": label
            })

        results.append(result_per_image)

    return {"results": results}
