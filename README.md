## CircuitGuard: PCB Defect Detection and Classification
### 🌟 Project Statement

The objective of the CircuitGuard project is to develop a fully automated system for defect detection and classification on Printed Circuit Boards (PCBs) using computer vision and deep learning techniques. The system will leverage reference-based image subtraction to localize defects and a deep Convolutional Neural Network (CNN) (EfficientNet-B4) for robust classification of the defect type.
The final deliverable is a functional web application that allows users to upload test and template PCB images and receive annotated results in real-time.

### ⚙️ Core Pipeline Overview

The system operates in a three-stage pipeline:

- **Subtraction Stage:** Detects potential defect regions by comparing a test PCB image against a defect-free template.
- **Extraction Stage:** Isolates individual defect areas using contour detection to create Regions of Interest (ROIs).
- **Classification Stage:** Predicts the type of defect (e.g., Short, Open, Missing Hole) using a trained CNN.


### 🛠️ Tech Stack

| Area              | Tools / Libraries     | Purpose                                                              |
|-------------------|------------------------|-----------------------------------------------------------------------|
| **Image Processing** | OpenCV, NumPy          | Alignment, Subtraction, Thresholding, and ROI Extraction.             |
| **Deep Learning**    | PyTorch, timm          | EfficientNet-B4 implementation, training, and inference.              |
| **Dataset**          | DeepPCB                | Source data for training and evaluation.                              |
| **Deployment**       | Streamlit / Python     | Frontend UI and modularized backend inference pipeline.               |


### 📁 Project Structure

CircuitGuard-PCB-Defect-Detection/
├── README.md               <-- This file
├── requirements.txt        <-- Python dependencies
├── data/
│   ├── raw/                <-- Downloaded DeepPCB image pairs (Template/Test)
│   └── processed/          <-- Labeled 128x128 defect ROI images (Output of M1)
├── src/
│   ├── M1_image_processing/
│   │   ├── subtraction.py  <-- Image subtraction and masking logic
│   │   └── roi_extraction.py <-- Contour detection and cropping logic
│   ├── M2_model_training/  <-- PyTorch scripts for EfficientNet (WIP)
│   └── web_app/            <-- Frontend and backend integration (WIP)
└── models/                 <-- Trained model checkpoints (WIP)



📈 Project Progress
Milestone 1: Image Processing (Complete)
Objective: Successfully generate a dataset of labeled defect Regions of Interest (ROIs) by localizing differences between template and test PCB images.
Module 1: Image Subtraction and Mask Generation
Task
Rationale / Technical Thinking
Image Alignment
The DeepPCB dataset provides mostly pre-aligned pairs. However, for real-world robustness, conversion to grayscale and a subsequent check for alignment (e.g., phase correlation or feature matching) is a necessary preprocessing step.
Absolute Difference
The core detection step. We use cv2.absdiff(test, template) to generate a difference map. This is crucial because a defect might be a missing copper line (darker than background) or an extra solder blob (lighter than background). The absolute difference ensures both types show up as high-intensity pixels.
Otsu Thresholding
The difference map contains varying intensity pixels. Instead of manually tuning a threshold value, Otsu's method is used to automatically find the optimal threshold that best separates the foreground (defects) from the background noise. This yields a clean, binary mask (cv2.THRESH_OTSU).

Module 2: Contour Detection and ROI Extraction
Task
Rationale / Technical Thinking
Morphological Operations
The binary mask is noisy. We apply a Closing Operation (cv2.MORPH_CLOSE), which is a combination of Dilation followed by Erosion. This operation helps to: 1) Fill small holes within genuine defect regions, and 2) Smooth the boundaries, resulting in more accurate bounding boxes.
Contour Extraction
cv2.findContours is used on the cleaned binary mask to identify the precise boundaries of each defect blob. This is more accurate than simply looking for pixel groups, as it provides geometric structure.
Bounding Box & Cropping
For each detected contour, cv2.boundingRect is used to define the minimal encompassing rectangle. This box is then used to crop the defect region from the original Test Image. Cropping the original image (not the mask) ensures the CNN receives all necessary color and texture information for accurate classification.
ROI Standardization
All extracted ROIs are resized to 128x128 pixels. This standardization is mandatory because CNNs like EfficientNet require fixed-size input tensors for batch processing.

➡️ Next Steps
The next focus is Milestone 2: Model Training and Evaluation. We will be implementing, training, and evaluating the EfficientNet-B4 model using the data/processed dataset generated in Milestone 1.
