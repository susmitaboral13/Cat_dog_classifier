from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model("cat_dog_classifier.h5")

def preprocess_image(img):
    img = np.array(img) / 255.0          # Normalize
    img = img.reshape(1, 150, 150, 3)    # Keep 4D tensor as model expects
    return img



@app.route('/')
def home():
    return render_template('index.html')

from io import BytesIO

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    if file:
        img_bytes = file.read()
        img = image.load_img(BytesIO(img_bytes), target_size=(150, 150))
        processed_img = preprocess_image(img)
        prediction = model.predict(processed_img)
        label = 'Dog' if prediction[0][0] > 0.5 else 'Cat'
        return render_template('index.html', prediction_text=f'Prediction: {label}')
    return 'No file uploaded', 400


if __name__ == "__main__":
    app.run(debug=True)
