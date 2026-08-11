from flask import Flask, render_template, request, send_from_directory, jsonify
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import sqlite3
from chatbot import get_chatbot_response

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "final_resnet50_balanced.pth")

classes = [
    'Alambadi', 'Amritmahal', 'Ayrshire', 'Banni', 'Bargur', 'Bhadawari',
    'Brown_Swiss', 'Dangi', 'Deoni', 'Gir', 'Guernsey', 'Hallikar',
    'Hariana', 'Holstein_Friesian', 'Jaffrabadi', 'Jersey', 'Kangayam',
    'Kankrej', 'Kasargod', 'Kenkatha', 'Kherigarh', 'Khillari',
    'Krishna_Valley', 'Malnad_gidda', 'Mehsana', 'Murrah', 'Nagori',
    'Nagpuri', 'Nili_Ravi', 'Nimari', 'Ongole', 'Pulikulam', 'Rathi',
    'Red_Dane', 'Red_Sindhi', 'Sahiwal', 'Surti', 'Tharparkar', 'Toda',
    'Umblachery', 'Vechur'
]

# Model

def load_model():
    model = models.resnet50(weights=None)

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(classes))

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model.to(device)
    model.eval()
    return model


print("Loading model...")
print("Model path:", MODEL_PATH)
print("Model exists:", os.path.exists(MODEL_PATH))

model = load_model()

print("Model loaded successfully!")

# Image Transformation

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Database

def get_breed_info(breed_name):

    db_path = os.path.join(BASE_DIR, "cows.db")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT origin, color, milk_yield, characteristics
        FROM breeds
        WHERE name = ?
        """,
        (breed_name,)
    )

    row = cursor.fetchone()

    conn.close()

    if row:
        return {
            "Origin": row[0],
            "Color": row[1],
            "Milk Yield": row[2],
            "Characteristics": row[3]
        }

    return {
        "Origin": "Unknown",
        "Color": "Unknown",
        "Milk Yield": "Unknown",
        "Characteristics": "No details found for this breed."
    }

# Prediction
def predict_breed(image_path):
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, predicted = torch.max(probs, 1)
        confidence = conf.item() * 100
        breed = classes[predicted.item()]

    if confidence < 35:
        return None, confidence

    return breed, round(confidence, 2)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict_page")
def predict_page():
    return render_template("predict.html")


@app.route("/predict", methods=["POST"])
def upload_and_predict():
    if "image" not in request.files:
        return jsonify({
            "error": "No image uploaded!"
        }), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({
            "error": "Empty file!"
        }), 400

    upload_folder = os.path.join(BASE_DIR, "uploads")
    os.makedirs(upload_folder, exist_ok=True)

    image_path = os.path.join(upload_folder, file.filename)
    file.save(image_path)

    breed, confidence = predict_breed(image_path)

    if breed is None:
        return jsonify({
            "warning": "This image does not appear to be a cow.",
            "confidence": round(confidence, 2)
        })

    info = get_breed_info(breed)
    return jsonify({
        "prediction": breed,
        "confidence": confidence,
        "info": info
    })

@app.route("/uploads/<filename>")
def send_file(filename):

    upload_folder = os.path.join(BASE_DIR, "uploads")

    return send_from_directory(
        upload_folder,
        filename
    )


# Chatbot

@app.route("/chatbot", methods=["POST"])
def chatbot():

    data = request.get_json()

    if not data or "message" not in data:
        return jsonify({
            "error": "No message provided!"
        }), 400

    user_message = data.get("message", "")
    image_data = data.get("image", None)

    response = get_chatbot_response(
        user_message,
        image_data
    )

    return jsonify({
        "response": response
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)