"""
This file will contain the code for graphical user interface (GUI). 
This module will interact with the user, get input (like image paths or camera feed), and display the results.
"""
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from camera import Camera
import cv2
import image_processor
from face_detector import FaceDetector

class Application:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.camera = Camera()
        self.face_detector = FaceDetector()

        # Canvas for video feed or images
        self.canvas_width = self.camera.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.canvas_height = self.camera.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.canvas = tk.Canvas(window, width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack()

        # Buttons
        self.btn_snapshot = tk.Button(window, text="Take Snapshot", width=50, command=self.snapshot)
        self.btn_snapshot.pack(anchor=tk.CENTER, expand=True)

        self.btn_load_image = tk.Button(window, text="Load Image", width=50, command=self.load_image)
        self.btn_load_image.pack(anchor=tk.CENTER, expand=True)

        self.window.mainloop()

    def snapshot(self):
        # Get a frame from the video source
        ret, frame = self.camera.get_frame()
        if ret:
            cv2.imwrite("frame-" + str(self.camera.cap.get(cv2.CAP_PROP_POS_FRAMES)) + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            self.process_and_display_image(frame)

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            image = cv2.imread(file_path)
            self.process_and_display_image(image)

    def process_and_display_image(self, image):
        # Resize the image to fit the canvas
        processed_image = image_processor.preprocess_image(image, self.canvas_width, self.canvas_height)

        faces = self.face_detector.detect_faces(processed_image)
        processed_image = image_processor.postprocess_image(processed_image, faces)
        self.display_image(processed_image)

    def display_image(self, image):
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

