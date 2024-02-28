from flask import Flask, request, jsonify
from PIL import Image
# import matplotlib.pyplot as plt
# import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import preprocessing
# import time
import pickle

app = Flask(__name__)
model = pickle.load(open('./EfficientNet_model.p', 'rb'))


def predict(image):
    IMAGE_SHAPE = (256, 256, 3)
    test_image = image.resize((256, 256))
    test_image = preprocessing.image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    class_names = [
        'Kari-Male',
        'Kathi',
        'Minukku-Female',
        'Pacha',
        'Red-beard',
        'White-beard']
    predictions = model.predict(test_image)
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()
    results = {
        'Kari-Male': 0,
        'Kathi': 1,
        'Minukku-Female': 2,
        'Pacha': 3,
        'Red-beard': 4,
        'White-beard': 5
    }

    result = f"{class_names[np.argmax(scores)]}"
    return result


@app.route('/classify', methods=['POST'])
def classify():
    if 'image' not in request.files:
        return jsonify({'error': 'no image found'})
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'no image found'})
    image = Image.open(image_file)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    prediction = predict(image)
    return jsonify({'prediction': prediction})


if __name__ == '__main__':
    app.run(debug=True)
