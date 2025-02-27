import os
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
import torch
from torchvision import transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
from groq import Groq  # Import Groq API
from flask_cors import CORS

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = Flask(__name__)
CORS(app)  # Allow all origins

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Dictionary of crop models and class labels
CROP_MODELS = {
    "Tomato": {
        "model_path": "models/plant_disease_resnet50.pth",
        "class_labels": [
            'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
            'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
            'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
            'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'
        ]
    },
    "Rice": {
        "model_path": "models/plant_disease_resnet50.pth",
        "class_labels": ["Rice_Bacterial_leaf_blight", "Rice_Brown_spot", "Rice_Leaf_Smut", "Rice_healthy"]
    },
    "Bell Pepper": {
        "model_path": "models/plant_disease_resnet50.pth",
        "class_labels": ["BellPepper_Bacterial_spot", "BellPepper_healthy"]
    },
    "Potato": {
        "model_path": "models/plant_disease_resnet50.pth",
        "class_labels": ["Potato_Early_blight", "Potato_Late_blight", "Potato_healthy"]
    },
    "Apple": {
        "model_path": "models/plant_disease_resnet50.pth",
        "class_labels": ["Apple_Black_rot", "Apple_Cedar_apple_rust", "Apple_healthy"]
    }
}

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model(crop):
    """Loads the correct model for the selected crop."""
    model_data = CROP_MODELS.get(crop)
    if not model_data:
        return None, None

    model = models.resnet50()
    num_features = model.fc.in_features
    num_classes = len(model_data["class_labels"])
    model.fc = nn.Sequential(
        nn.Dropout(0.01),
        nn.Linear(num_features, num_classes)
    )
    
    model.load_state_dict(torch.load(model_data["model_path"], map_location=torch.device("cpu")))
    model.eval()
    return model, model_data["class_labels"]


def predict_image(image_path, model, class_labels):
    """Predicts the class of an image using the selected model."""
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)
        predicted_class_index = torch.argmax(probabilities, dim=1).item()
        predicted_class = class_labels[predicted_class_index]
        confidence = probabilities[0][predicted_class_index].item()

    return predicted_class, confidence


def query_llm(disease_name, confidence):
    """Query Groq Llama model for remedies based on the predicted disease."""
    try:
        client = Groq(api_key=GROQ_API_KEY)
        query = f"Provide remedies, causes, and relevant details for {disease_name} with confidence {confidence:.2f}."
        messages = [{"role": "user", "content": [{"type": "text", "text": query}]}]

        response = client.chat.completions.create(
            messages=messages, 
            model="llama-3.2-90b-vision-preview"
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error fetching remedies: {str(e)}"


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        crop = request.form.get("crop")
        if crop not in CROP_MODELS:
            return render_template('index.html', message="Invalid crop selection", crops=CROP_MODELS.keys())

        if 'file' not in request.files:
            return render_template('index.html', message='No file part', crops=CROP_MODELS.keys())

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', message='No selected file', crops=CROP_MODELS.keys())

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Load the selected crop model
            model, class_labels = load_model(crop)
            if model is None:
                return render_template('index.html', message='Error loading model', crops=CROP_MODELS.keys())

            # Make prediction
            predicted_class, confidence = predict_image(filepath, model, class_labels)

            # Query Groq Llama model
            disease_info = query_llm(predicted_class, confidence)

            return render_template(
                'index.html',
                message=f'Predicted class: {predicted_class}, Confidence: {confidence:.4f}',
                disease_info=disease_info,
                image_path=filepath,
                crops=CROP_MODELS.keys()
            )
        else:
            return render_template('index.html', message='Allowed image types: png, jpg, jpeg', crops=CROP_MODELS.keys())

    return render_template('index.html', message='', crops=CROP_MODELS.keys())


@app.route('/uploads/<filename>')
def send_image(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8000)
