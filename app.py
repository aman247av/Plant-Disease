import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pickle
from flask import Flask, request, jsonify
import os
import io

model_path = 'model/trained_model.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_class_labels(file_path):
    with open(file_path, "r") as file:
        class_labels = [line.strip() for line in file.readlines()]
    return class_labels

class_labels = load_class_labels('Plant-Disease\class_labels.txt')

app = Flask(__name__)

def predict_image(image, model, transform, device, class_labels):
    image = Image.open(io.BytesIO(image)).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        predicted_class = class_labels[predicted.item()]
        confidence_score = confidence.item() * 100
    
    return predicted_class, confidence_score

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    image = request.files['image'].read()
    predicted_class, confidence_score = predict_image(image, model, transform, device, class_labels)
    
    return jsonify({
        'predicted_class': predicted_class,
        'confidence_score': f"{confidence_score:.2f}%"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
