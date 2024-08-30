import cv2
import face_detection
from PIL import Image
import numpy as np
import tensorflow

model = tensorflow.keras.models.load_model('./kathakali.h5')

detector = face_detection.build_detector(
    "DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)
emotion = ['Anger', 'Love', 'Odious', 'Pitty',
           'Terrible', 'Peace', 'Comic', 'Heroic', 'Wonder']


# def predict(image):
#     img = np.asarray(image)
#     faces = detector.detect(img)
#     print(f'face found: {len(faces)}')
#     faces = faces.astype(int)
#     # loop over all detected faces
#     accu = [0, 0, 0, 0, 0, 0, 0, 0, 0]
#     if len(faces) > 0:
#         for x, y, w, h, prob in faces:
#             try:
#                 face = img[y:h, x:w]
#                 eiei = cv2.resize(face, (100, 100))
#                 cv2.imshow("Cropped Face", eiei)
#                 cv2.waitKey(0)

#                 new = cv2.resize(face, (32, 32))
#                 new = new.reshape(1, 32, 32, 3)
#                 new = new.astype('float32')
#                 dummy = new.copy() / 255
#                 pred = model.predict(dummy)
#                 i = pred.squeeze()
#                 for number in range(9):
#                     print(f'{emotion[number]}: {round(i[number] * 100, 2)} %')
#                 print(f'This picture is {emotion[np.argmax(i)]}')
#                 accu[np.argmax(i)] += 1
#             except:
#                 print('error occurred')
#     return "test"


# with Image.open("1.jpg") as im:
#     # im.show()
#     predict(im)
