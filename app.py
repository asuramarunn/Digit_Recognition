import base64
from io import BytesIO
from flask import Flask, render_template, request, redirect, url_for, flash
import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError
from preprocessing import sharpening_img, extract_character
from cnn_model import create_cnn_model
import tensorflow as tf

# Initialize Flask app and model
app = Flask(__name__)
app.secret_key = "your_secret_key"

# Load the model and set up TensorFlow
model = create_cnn_model()
model.load_weights('cnn_model_weights.weights.h5')
tf.config.run_functions_eagerly(True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file:
        try:
            # Read the image file with OpenCV
            np_img = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE)
            
            # Step 1: Sharpen the image
            sharpened_img = sharpening_img(img)

            # Step 2: Extract the character using the extract_character function
            character_img = extract_character(sharpened_img)

            # Step 3: Resize the cropped character to 28x28 for prediction
            resized_img = cv2.resize(character_img, (28, 28))
            img_np = resized_img.reshape(1, 28, 28, 1)  # Reshape for model input

            # Predict
            prediction = model.predict(img_np)
            predicted_class = np.argmax(prediction)

            # Step 4: Draw bounding box on the original sharpened image
            # Find the bounding box of the extracted character
            x, y, w, h = cv2.boundingRect(cv2.findContours(cv2.threshold(sharpened_img, 128, 255, cv2.THRESH_BINARY_INV)[1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0])

            # Draw the bounding box on the sharpened image
            color_img = cv2.cvtColor(sharpened_img, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(color_img, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Red bounding box

            # Encode the image with bounding box to base64 for HTML rendering
            _, buffer = cv2.imencode('.png', color_img)
            image_data = base64.b64encode(buffer).decode('utf-8')

            return render_template(
                'index.html',
                prediction_text=f'Predicted Digit: {predicted_class}',
                image_data=image_data
            )
        
        except UnidentifiedImageError:
            flash("Invalid file format. Please upload an image file.")
            return redirect(request.url)


@app.route('/reset', methods=['POST'])
def reset():
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
