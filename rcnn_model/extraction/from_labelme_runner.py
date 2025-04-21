import os
import labelme2coco
from PIL import Image
from from_root import from_root
import floorplan_sampler
import json


def main():
    convert_from_labelme()

    handle_jpeg_files("annotations/train.json")
    handle_jpeg_files("annotations/val.json")

    print("NOW FOR SAMPLING")
    floorplan_sampler.main()

def convert_from_labelme():
    os.chdir(str(from_root("dataset")))
    labelme_source_dir = "labelme_data"
    annotation_dest_dir = "annotations"
    training_split_percentage = .8
    labelme2coco.convert(labelme_source_dir, annotation_dest_dir, training_split_percentage, category_id_start=0)


def handle_jpeg_files(coco_path):
    #open file
    file = open(coco_path,"r+")
    coco = json.load(file)

    #find and edit jpeg images
    for image in coco["images"]:
        img_name = image["file_name"]
        if(".jpg" in img_name or ".jpeg" in img_name):
            new_img_name = convert_to_png(img_name)
            image["file_name"]=new_img_name

    #save
    file.seek(0)
    json.dump(coco, file, indent="  ")
    file.close()


def convert_to_png(img_path):
    #load image
    img = Image.open(img_path)

    #remove .jpg or .jpeg from path
    if(".jpeg" in img_path):
        img_path = img_path[0:-5]
    else:
        img_path = img_path[0:-4]
    
    #add .png and save
    img_path += ".png"
    img.save(img_path)
    return img_path



main()
