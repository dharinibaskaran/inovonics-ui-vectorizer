# single_image_cleaning.py

import cv2
import numpy as np
import os

def preprocess_image(image_path):
    """
    Preprocess a single floorplan image: denoising, CLAHE, edge enhancement.
    """
    print(f"🧹 Preprocessing image: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Could not read image from {image_path}")
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    denoisy_img = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoisy_img)

    # Apply threshold
    _, thresholded = cv2.threshold(enhanced, 150, 255, cv2.THRESH_BINARY)

    # Detect edges
    edges = cv2.Canny(thresholded, 100, 220, apertureSize=3)

    # Detect lines and draw them
    output_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=50,
                            minLineLength=35, maxLineGap=5)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(output_img, (x1, y1), (x2, y2), (210, 210, 210), 1)

    # Blend the original image with line-enhanced version
    blended_image = cv2.addWeighted(image, 0.7, output_img, 0.3, 0)

    print(f"✅ Preprocessing complete for: {image_path}")
    return blended_image
