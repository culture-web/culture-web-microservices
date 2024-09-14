import face_detection
import cv2
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import base64


app = Flask(__name__)

detector = face_detection.build_detector(
    "DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)


def encode_image(image):
    """Encode image as a Base64 string."""
    _, buffer = cv2.imencode('.jpg', image)
    encoded_string = base64.b64encode(buffer).decode('utf-8')
    return encoded_string


def predict(image):
    img = np.asarray(image)
    faces = detector.detect(img)
    print(f'face found: {len(faces)}')
    faces = faces.astype(int)
    detected_faces = []
    if len(faces) > 0:
        for x, y, w, h, prob in faces:
            try:
                face = img[y:h, x:w]
                detected_faces.append(face)
            except:
                print('error occurred')
    return detected_faces, faces


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
    detected_faces, locations = predict(image)
    encoded_faces = [encode_image(face) for face in detected_faces]
    return jsonify({'faces': encoded_faces, 'locations': locations.tolist()})


if __name__ == '__main__':
    app.run()
