import os
import random
import shutil

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

print(f"Fixed Working Directory: {os.getcwd()}")

dataset_path = "dataset"
image_dir = os.path.join(dataset_path, "images")
label_dir = os.path.join(dataset_path, "yolo_annotations")

train_dir = "dataset/train"
val_dir = "dataset/val"
test_dir = "dataset/test"

for d in [train_dir, val_dir, test_dir]:
    os.makedirs(os.path.join(d, "images"), exist_ok=True)
    os.makedirs(os.path.join(d, "yolo_annotations"), exist_ok=True)

images = [f for f in os.listdir(image_dir) if f.endswith(".png")]
random.shuffle(images)

split_ratio = [0.8, 0.1, 0.1]
train_split = int(split_ratio[0] * len(images))
val_split = int(split_ratio[1] * len(images)) + train_split

train_images = images[:train_split]
val_images = images[train_split:val_split]
test_images = images[val_split:]

for img in train_images:
    shutil.move(os.path.join(image_dir, img), os.path.join(train_dir, "images", img))
    shutil.move(os.path.join(label_dir, img.replace(".png", ".txt")), os.path.join(train_dir, "yolo_annotations", img.replace(".png", ".txt")))

for img in val_images:
    shutil.move(os.path.join(image_dir, img), os.path.join(val_dir, "images", img))
    shutil.move(os.path.join(label_dir, img.replace(".png", ".txt")), os.path.join(val_dir, "yolo_annotations", img.replace(".png", ".txt")))

for img in test_images:
    shutil.move(os.path.join(image_dir, img), os.path.join(test_dir, "images", img))
    shutil.move(os.path.join(label_dir, img.replace(".png", ".txt")), os.path.join(test_dir, "yolo_annotations", img.replace(".png", ".txt")))

print("Dataset split completed!")
