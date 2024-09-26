from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('/home/omar/Desktop/AISE/Work/26-8-2024 Learn Phase/CIFAR/Deployment/cifar10.keras')

# List of category names for CIFAR-10 dataset
category_names = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    # If the user does not select a file, the browser submits an empty file without a filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Read and preprocess the image file
        img = Image.open(file.stream).convert('RGB')  # Ensure RGB format
        img = img.resize((32, 32))  # Resize to 32x32 pixels
        img_array = np.array(img) / 255.0  # Normalize the image
        img_array = img_array.reshape((1, 32, 32, 3))  # Reshape for the model
        
        # Make prediction
        prediction = model.predict(img_array)
        
        # Get the class with the highest probability
        predicted_class = np.argmax(prediction, axis=1)[0]
        category_name = category_names[predicted_class]
        
        # Return the result as JSON
        return jsonify({'prediction': category_name})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
