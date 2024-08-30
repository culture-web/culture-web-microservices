import cv2
import face_detection
from flask import Flask, request, jsonify
from PIL import Image
import tensorflow
import numpy as np

app = Flask(__name__)
model = tensorflow.keras.models.load_model('./kathakali.h5')

detector = face_detection.build_detector(
    "DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)
emotion = ['Anger', 'Love', 'Odious', 'Pitty',
           'Terrible', 'Peace', 'Comic', 'Heroic', 'Wonder']


def predict(image):
    img = np.asarray(image)
    faces = detector.detect(img)
    print(f'face found: {len(faces)}')
    faces = faces.astype(int)
    # loop over all detected faces
    accu = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    if len(faces) > 0:
        for x, y, w, h, prob in faces:
            try:
                face = img[y:h, x:w]
                eiei = cv2.resize(face, (100, 100))
                cv2.imshow("Cropped Face", eiei)
                new = cv2.resize(face, (32, 32))
                new = new.reshape(1, 32, 32, 3)
                new = new.astype('float32')
                dummy = new.copy() / 255
                pred = model.predict(dummy)
                i = pred.squeeze()
                print(face)
                print(pred)
                print(i)
                for number in range(9):
                    print(f'{emotion[number]}: {round(i[number] * 100, 2)} %')
                print(f'This picture is {emotion[np.argmax(i)]}')
                accu[np.argmax(i)] += 1
            except:
                print('error occurred')
    return "test"


@ app.route('/classify', methods=['POST'])
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
    app.run()
