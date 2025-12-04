import cv2
import numpy as np
def extract_contours(mask):
    # Noise Reduction: Use morphological operations (e.g., closing or opening)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # Helps close small holes/gaps

    # Find contours
    # cv2.RETR_EXTERNAL retrieves only the outer contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def extract_rois(original_test_img, contours, defect_label):
    roi_list = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)

        # Crop the original test image
        roi = original_test_img[y:y+h, x:x+w]

        # Resize for model training (128x128 as specified)
        resized_roi = cv2.resize(roi, (128, 128))

        # Save the labeled ROI
        filename = f"defect_{i}_{defect_label}.png"
        cv2.imwrite(f"data/processed/{filename}", resized_roi)
        roi_list.append(resized_roi)
    return roi_list