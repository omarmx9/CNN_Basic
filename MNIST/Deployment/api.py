from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('MNIST.keras')

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
    
    # Read the image file
    img = Image.open(file.stream).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28 pixels
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = img_array.reshape((1, 28, 28, 1))  # Reshape for the model
    
    # Make prediction
    prediction = model.predict(img_array)
    
    # Get the class with the highest probability
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    # Return the result as JSON
    return jsonify({'prediction': int(predicted_class)})

if __name__ == '__main__':
    app.run(debug=True)