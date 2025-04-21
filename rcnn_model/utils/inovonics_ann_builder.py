import json
import numpy as np
from datetime import datetime
import random

version_number = "0.0.1"

class InovonicsAnnotationBuilder:
    #creates the base structure of the coco format
    def __init__(self):
        self.annotation_id = ""
        self.body = []
        self.type = ""
        self.target = {}

    def set_id(self, id):
        self.annotation_id = id

    def set_body(self, room_name, locator_value=None):
        color = self.generate_color()
        self.body = [{"type":"TextualBody",
                     "value":room_name},
                     {"type":"highlighting",
                     "value":color},
                     {"type":"locators",
                     "value":locator_value}]
        
    def generate_color(self):
        red = str(random.randint(0,255))
        green = str(random.randint(0,255))
        blue = str(random.randint(0,255))
        return "rgb("+red+","+green+","+blue+")"

    def set_type(self, type):
        self.type = type

    def set_target(self, type, url, bbox):
        rect = self.bbox_to_rect(bbox)
        self.target = {"selector":{"type":type,
                                 "conformsTo":url,
                                 "value":rect}}
    
    def bbox_to_rect(self,bbox):
        return "xywh=pixel"+str(bbox[0])+","+str(bbox[1])+","+str(bbox[2])+","+str(bbox[3])
    
    def final_output(self):
        return {"annotation_id":self.annotation_id, "body":self.body, "type":self.type, "target":self.target}
    
    def save_file(self, filepath):
        coco_file = open(filepath,'w')
        json.dump(self.final_output(),coco_file,indent=4)
        coco_file.close()