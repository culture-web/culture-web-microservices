import cv2
from flask import Flask, request, jsonify
from PIL import Image
import tensorflow
import numpy as np

app = Flask(__name__)
model = tensorflow.keras.models.load_model('./kathakali.h5')

emotion = ['Anger', 'Love', 'Odious', 'Pitty',
           'Terrible', 'Peace', 'Comic', 'Heroic', 'Wonder']


def predict(image):
    img = np.asarray(image)
    # Loop through the img and predict the face expression
    try:
        new = cv2.resize(img, (32, 32))
        new = new.reshape(1, 32, 32, 3)
        new = new.astype('float32')
        dummy = new.copy() / 255
        pred = model.predict(dummy)
        i = pred.squeeze()
        accu = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        for number in range(9):
            accu[number] = round(i[number] * 100, 2)
        prediction = emotion[np.argmax(i)]
    except:
        prediction = 'error occurred'
        accu = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    return {'prediction': prediction, 'accuracy': accu}

@ app.route('/expression', methods=['POST'])
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
    return jsonify(prediction)


if __name__ == '__main__':
    app.run()
