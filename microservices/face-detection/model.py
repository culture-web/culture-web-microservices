from retinaface import RetinaFace
import numpy as np

RetinaFace.detect_faces(np.zeros((256, 256, 3), dtype=np.uint8))