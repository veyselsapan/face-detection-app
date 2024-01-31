"""
This module handles image processing tasks. 
"""
import cv2
import numpy as np

def preprocess_image(image, target_width, target_height):
    """
    Preprocess the image before face detection.
    This function will resize the image to fit the specified dimensions.
    :param image: Input image as a numpy array
    :param target_width: The width of the target size
    :param target_height: The height of the target size
    :return: Preprocessed image
    """
    # Convert target dimensions to integers
    target_width = int(target_width)
    target_height = int(target_height)

    # Resize image to fit the target size while maintaining aspect ratio
    height, width = image.shape[:2]
    scale = min(target_width / width, target_height / height)

    new_width = int(width * scale)
    new_height = int(height * scale)

    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Create a new image and place the resized image within it
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2
    canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_image

    return canvas

def postprocess_image(image, faces):
    """
    Postprocess the image after face detection.
    This can include drawing bounding boxes, applying filters, etc.
    :param image: Image as a numpy array
    :param faces: List of detected faces with their coordinates
    :return: Postprocessed image
    """
    # Draw bounding boxes around detected faces
    for x, y, width, height in faces:
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)

    return image
