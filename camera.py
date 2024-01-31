"""
This module will handle camera operations. 
It is responsible for capturing video frames, interfacing with the face_detector module, and displaying the processed video stream.
"""
import cv2
from face_detector import FaceDetector
import image_processor  # Import the image_processor module

class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)  # 0 is usually the default camera
        self.face_detector = FaceDetector()

    def get_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Detect faces
            faces = self.face_detector.detect_faces(frame)

            # Postprocess the image (e.g., draw bounding boxes)
            processed_frame = image_processor.postprocess_image(frame, faces)

        return ret, processed_frame

    def release(self):
        self.cap.release()
