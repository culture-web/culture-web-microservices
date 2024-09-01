import face_detection
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image

app = Flask(__name__)

detector = face_detection.build_detector(
    "DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)


def predict(image):
    img = np.asarray(image)
    faces = detector.detect(img)
    print(f'face found: {len(faces)}')
    faces = faces.astype(int)
    return faces


@ app.route('/detect', methods=['POST'])
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

    return jsonify(prediction.tolist())


if __name__ == '__main__':
    app.run()
