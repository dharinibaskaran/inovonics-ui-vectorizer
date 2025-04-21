import json
import numpy as np
from datetime import datetime

version_number = "0.0.1"

class AnnotationBuilder:
    #creates the base structure of the coco format
    def __init__(self):
        self.licenses = []
        self.categories = [{"id": 0,
                             "name": "Wall",
                             "supercategory": "none"},
                           { "id": 1,
                             "name": "Door",
                             "supercategory": "none"},
                           {"id": 2,
                             "name": "Room",
                             "supercategory": "none"},
                           {"id": 3,
                            "name": "Window",
                            "supercategory": "none"}
                          ]
        self.images = []
        self.annotations = []

    def set_info(self, description, data_source_name, data_source_url, data_source_creation_date):
        self.info = [{"year":2025,
                        "version":version_number,
                        "description":description,
                        "contributor":data_source_name,
                        "url":data_source_url,
                        "date_created":data_source_creation_date.strftime("%Y-%m-%dT%H:%M:%S")}]

    def add_license(self, license_name, license_url):
        self.licenses.append({"id":len(self.licenses),
                          "url":license_url,
                          "name":license_name})

    def add_image(self, filename, width, height):
        id = len(self.images)
        self.images.append({"id":id,
                            "width":width,
                            "height":height,
                            "file_name":filename, #filename should be the image path relative to the cocofile's path
                            "license":0,
                            "date_captured":datetime.now().strftime("%Y-%m-%dT%H:%M:%S")})
        return id

    def add_annotation(self, image_id, category_id, poly):
        id = len(self.annotations)
        segmentation = np.array(poly.exterior.coords).astype(int).ravel().tolist()[:-2]
        x,y,x2,y2 = tuple(map(int, poly.bounds))
        self.annotations.append({"id":id,
                                 "image_id":image_id,
                                 "category_id":category_id,
                                 "segmentation":[segmentation], 
                                 "area":poly.area,
                                 "bbox":[x,y,x2-x,y2-y],
                                 "iscrowd":0})
        return id, poly
    
    def final_output(self):
        return {"info":self.info, "licenses":self.licenses, "categories":self.categories, "images":self.images, "annotations":self.annotations}
    
    def save_file(self, filepath):
        coco_file = open(filepath,'w')
        json.dump(self.final_output(),coco_file,indent=4)
        coco_file.close()