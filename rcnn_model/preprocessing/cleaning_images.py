import cv2
import numpy as np
import os

# Function to preprocess the image
def preprocessing(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print(f"Warning: Could not read {image_path}")
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoisy_img = cv2.GaussianBlur(gray, (5, 5), 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(denoisy_img)

    _, thresholded_img = cv2.threshold(clahe_img, 150, 255, cv2.THRESH_BINARY)
    
    edges = cv2.Canny(thresholded_img, 100, 220, apertureSize=3)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=50,
                            minLineLength=35, maxLineGap=5)

    output_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(output_img, (x1, y1), (x2, y2), (210, 210, 210), 1)

    blended_image = cv2.addWeighted(image, 0.7, output_img, 0.3, 0)

    return blended_image

# Define paths
source_root = "../cubicasa5k"
output_dir = "dataset/images"

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

print(f"Fixed Working Directory: {os.getcwd()}")

# Create output directories if they don't exist
os.makedirs("dataset", exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Iterate over subfolders 1, 2, 3, ..., n
for subfolder in os.listdir(source_root):
    subfolder_path = os.path.join(source_root, subfolder)

    if os.path.isdir(subfolder_path):  # Ensure it's a directory
        image_path = os.path.join(subfolder_path, "F1_original.png")

        if os.path.exists(image_path):
            processed_img = preprocessing(image_path)

            if processed_img is not None:
                output_filename = f"{subfolder}.png"  # Save with subfolder name
                output_path = os.path.join(output_dir, output_filename)
                cv2.imwrite(output_path, processed_img)
                print(f"Processed: {image_path} -> {output_path}")
        else:
            print(f"Skipping {subfolder}: F1_original.png not found")

print("Processing completed.")
