from flask import Flask, request, jsonify
import joblib
import numpy as np
from PIL import Image
import cv2
import os

app = Flask(__name__)

# Get the absolute path of parent directory
base_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)
model_dir = os.path.join(base_dir, "models")
dump_dir = os.path.join(base_dir, "dump")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(dump_dir, exist_ok=True)

# Load the model using joblib
model_path = os.path.join(model_dir, "best_model.pkl")
model = joblib.load(model_path)

@app.route("/predict_digit", methods=["POST"])
def predict():
    """Accepts an image file and predicts the digit."""
    # Get the image from the POST request
    file = request.files['image']
    image_path = os.path.join(dump_dir, file.filename)
    file.save(image_path)
    
    # Preprocess the image
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = np.array(image)  # Convert image to numpy array
    image = cv2.resize(image, (8, 8))  # Resize image to 8x8, as in the Digits dataset
    image = (16 * (image / 255.0)).astype(np.uint8)  # Scale pixel values to range [0, 16]
    image = image.flatten().reshape(1, -1)  # Flatten and reshape to match model input
    
    
    # Predict the digit
    prediction = model.predict(image)
    
    # Return the prediction as JSON
    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
