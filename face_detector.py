"""
This file contains the core logic for face detection. 
It uses MTCNN model for detecting faces in images. 
This module will be responsible for loading the model and returning the detection results.
"""

from mtcnn import MTCNN

class FaceDetector:
    def __init__(self):
        # Initialize the MTCNN detector
        self.detector = MTCNN()

    def detect_faces(self, image):
        # Detect faces
        faces = self.detector.detect_faces(image)
        # Extract bounding box for each face
        face_boxes = [face['box'] for face in faces]
        return face_boxes

