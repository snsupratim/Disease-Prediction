import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import torch
from torchvision import transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder  # Import LabelEncoder

app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create the folder if it doesn't exist

# Model Path (Update this with your actual model path)
MODEL_PATH = 'plant_disease_resnet50.pth'

# Class labels - IMPORTANT:  These MUST match your training order.  This is critical.
class_labels = ['Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold',
                'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
                'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']


# 1. Load the Model (Load it *once* when the app starts)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50()
num_features = model.fc.in_features
num_classes = len(class_labels)
model.fc = nn.Sequential(
    nn.Dropout(0.01),
    nn.Linear(num_features, num_classes)
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device)) # Load saved state dictionary
model.to(device)
model.eval()  # Set to evaluation mode
print("Model loaded successfully!")  # Add a print statement to confirm


# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Use if you normalized during training
])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_image(image_path):
    """Predicts the class of an image using the loaded model."""
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class_index = torch.argmax(probabilities, dim=1).item()
        predicted_class = class_labels[predicted_class_index]
        confidence = probabilities[0][predicted_class_index].item()

    return predicted_class, confidence


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            return render_template('index.html', message='No selected file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Make prediction
            predicted_class, confidence = predict_image(filepath)
            message = f'Predicted class: {predicted_class}, Confidence: {confidence:.4f}'
            return render_template('index.html', message=message, image_path=filepath) # Pass the image path to display

        else:
            return render_template('index.html', message='Allowed image types are -> png, jpg, jpeg')

    return render_template('index.html', message='')


@app.route('/uploads/<filename>')
def send_image(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == '__main__':
    from flask import send_from_directory
    app.run(debug=True,host="0.0.0.0", port=8000)
