import os
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from torchvision import transforms, models
from torch import nn
from PIL import Image
from fastapi.responses import FileResponse
import aiofiles
import requests

app = FastAPI()

# Configure upload folder
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Model path & class labels
MODEL_PATH = "plant_disease_resnet50.pth"
class_labels = [
    "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight", "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot", "Tomato_Spider_mites_Two_spotted_spider_mite", "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus", "Tomato__Tomato_mosaic_virus", "Tomato_healthy"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50()
num_features = model.fc.in_features
num_classes = len(class_labels)
model.fc = nn.Sequential(nn.Dropout(0.01), nn.Linear(num_features, num_classes))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

GROQ_API_KEY = "your_groq_api_key"  # Replace with your actual API key
GROQ_API_URL = "https://api.groq.com/v1/chat/completions"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class_index = torch.argmax(probabilities, dim=1).item()
        predicted_class = class_labels[predicted_class_index]
        confidence = probabilities[0][predicted_class_index].item()
    
    return predicted_class, confidence

def fetch_groq_info(disease_name):
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    data = {"model": "llama3", "messages": [{"role": "system", "content": f"Provide causes and remedies for {disease_name}."}]}
    response = requests.post(GROQ_API_URL, json=data, headers=headers)
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    return "Error fetching information from Groq API."

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PNG, JPG, JPEG allowed.")
    
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    async with aiofiles.open(file_path, "wb") as out_file:
        content = await file.read()
        await out_file.write(content)
    
    disease, confidence = predict_image(file_path)
    remedies = fetch_groq_info(disease)
    
    return {"disease": disease, "confidence": confidence, "remedies": remedies, "image_url": f"/uploads/{file.filename}"}

@app.get("/uploads/{filename}")
async def get_image(filename: str):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="File not found.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
