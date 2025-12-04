import cv2
import numpy as np

def create_difference_map(test_path, template_path):
    # Load images
    test_img = cv2.imread(test_path)
    template_img = cv2.imread(template_path)

    # Convert to grayscale
    test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)

    # Image Subtraction: Calculate absolute difference
    # Alignment step (omitted for DeepPCB as images are usually pre-aligned)
    # If alignment is needed, use cv2.matchTemplate or feature matching first.
    diff = cv2.absdiff(test_gray, template_gray)
    return diff

def create_defect_mask(diff_map):
    # Otsu Thresholding (cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # This automatically finds the optimal threshold to separate foreground (defects) from background.
    _, mask = cv2.threshold(diff_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask