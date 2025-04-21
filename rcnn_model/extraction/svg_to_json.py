from xml.dom import minidom
import cv2
import numpy as np
import math
from annotation_builder import AnnotationBuilder as AnnBuild
from pycocotools.coco import COCO
from shapely import geometry
from datetime import datetime
import os
import sys
import random
from from_root import from_root
from rcnn_model.preprocessing.cleaning_single_image import preprocess_image
from rcnn_model.utils.floorplan_vectorizer_utils import get_image_size, draw_from_coco



# sys.path.append(str(from_root("preprocessing")))
# from cleaning_images import preprocessing
# sys.path.append(str(from_root("utils")))
# from floorplan_vectorizer_utils import get_image_size, draw_from_coco

### After running, its split with https://github.com/akarazniewicz/cocosplit
### This may or may not be temporary

### Main functionality ###

scale_factor = .5
dataset_root = str(from_root("dataset"))+"/"

def main():
    extract_all_cubicasa_anns(True)


def extract_all_cubicasa_anns(export_image=False):
    #initialize annotation builder
    ann_builder = AnnBuild()
    ann_builder.set_info("converted from cubicasa 5k SVG file","cubicasa 5k","https://github.com/cubicasa/cubicasa5k",datetime(2019,5,24))
    ann_builder.add_license("Creative Commons Attribution-NonCommercial 4.0 International License", "http://creativecommons.org/licenses/by-nc/4.0/")
    #iterate through cubicasa files
    for name in os.listdir(str(from_root(dataset_root+"cubicasa_data/"))):
        process_cubicasa_image(ann_builder, name)
    #save data
    print("SAVING TO annotations/cubicasa_coco.json")
    ann_builder.save_file(str(from_root(dataset_root+"annotations/cubicasa_coco.json")))
    if(export_image):
        save_validation_images(str(from_root(dataset_root+"annotations/cubicasa_coco.json")))


def process_cubicasa_image(ann_builder, name):
    #load and preprocess image
    print("\nprocessing "+name)
    source_img_path = str(from_root(dataset_root+"cubicasa_data/"+name+"/F1_scaled.png"))
    processed_img_path = str(from_root(dataset_root+"preprocessed/casa"+name+".png"))
    apply_preprocessing(source_img_path, processed_img_path)

    #load svg
    source_svg_path = str(from_root(dataset_root+"cubicasa_data/"+name+"/model.svg"))
    print("from "+source_svg_path)
    print("image in "+processed_img_path)

    #extract data from svg
    try:
        width, height = get_image_size(processed_img_path)
        ann_builder = process_cubicasa(ann_builder, source_svg_path, processed_img_path, width, height)
    except:
        print("ERROR while extracting "+name)
        print(sys.exc_info())


def find_svg(path, name):
    for file in os.listdir(path):
        found_name = file.startswith(name+"_gt_")
        if(found_name):
            found_svg = file.endswith(".svg")
            if(found_svg):
                return path+file


def process_cubicasa(ann_builder, sourve_svg_path, source_img_path, width, height):
    #Get points
    doc = minidom.parse(sourve_svg_path)
    walls = extract_casa_elements_with_id("Wall",doc)
    windows = extract_casa_elements_with_id("Window",doc)
    doors = extract_casa_elements_with_id("Door",doc)
    doc.unlink()
    #export to JSON and potentially imges for visual confirmation that the process works
    ann_builder = export_to_builder_casa(ann_builder,source_img_path,width,height,walls,doors,windows)
    return ann_builder


### Coco Formatting/Export ###

def export_to_builder_casa(ann_builder,source_img,width,height,walls,doors,windows):
    #initialization
    id = ann_builder.add_image(source_img, width, height)
    #walls
    wall_polygons = get_features_from_ann_set(walls)
    door_polygons = get_features_from_ann_set(doors)
    window_polygons = get_features_from_ann_set(windows)
    features = wall_polygons + door_polygons + window_polygons
    rooms = create_rooms_from_features(features, width, height)
    for poly in rooms.geoms:
        ann_builder.add_annotation(id, 2, poly)
    return ann_builder


def get_features_from_ann_set(set, coco = None, image_id = 0, category_id = 0):
    polygons = []
    for points in set:
        poly = geometry.Polygon([[p[0], p[1]] for p in points])
        if(coco is not None):
            coco.add_annotation(image_id, category_id, poly)
        polygons.append(poly)
    return polygons


def create_rooms_from_features(features, width, height):
    room_polygons = geometry.Polygon([(0,0),
                                    (width,0),
                                    (width,height),
                                    (0,height)
                                    ])
    for poly in features:
        room_polygons = room_polygons.difference(poly,3)
    return geometry.MultiPolygon(room_polygons.geoms[1:]) #this eliminates the exterior from the rooms


def apply_preprocessing(source_path, processed_path):
    img = cv2.imread(source_path)
    small_img = cv2.resize(img, (0,0), fx=scale_factor, fy=scale_factor)
    cv2.imwrite(processed_path,small_img)
    processed_img = preprocess_image(processed_path)
    #small = cv2.resize(processed, (0,0), fx=scale_factor, fy=scale_factor)
    cv2.imwrite(processed_path,processed_img)
    print(get_image_size(source_path))
    print(get_image_size(processed_path))


### SVG element extraction ###

def get_casa_size(doc):
    path = doc.getElementsByTagName('svg')[0]
    return int(float(path.getAttribute('width'))), int(float(path.getAttribute('height')))


def extract_casa_elements_with_id(id, doc):
    elements = []
    for path in doc.getElementsByTagName('g'):
        #iterates through everything and finds items labelled as walls
        if(id in path.getAttribute('id')):
            #luckily, the first attribute after all of these is a polygon containing a list of coordinate points
            string = path.firstChild.getAttribute('points')
            points = points_string_to_int_points(string)
            elements.append(points)
    return elements


### Helper Functions ###

def quadrilateral_to_line(points):
    base_point = [0,0]
    points.sort(key=lambda p: check_distance(base_point,p))
    base_point = points[0]
    points.sort(key=lambda p: check_distance(base_point,p))
    return np.array([get_midpoint(points[0], points[1]), get_midpoint(points[2], points[3])])


def check_distance(point_A, point_B):
    return math.sqrt(((point_A[0]-point_B[0]) ** 2) + ((point_A[1]-point_B[1]) ** 2))


def get_midpoint(point_A, point_B):
    return np.array([round((point_A[0]+point_B[0])/2), round((point_A[1]+point_B[1])/2)])


def points_string_to_int_points(string):
    return [[int(round(float(pi)*scale_factor)) for pi in p.split(",")] for p in string.split()]


### Validation Images ###

def save_validation_images(filepath):
    count = 0
    result = COCO(filepath)
    for id in random.sample(result.getImgIds(), 15):
        print("IMAGE "+str(result.imgs[id]))
        validation_path = str(from_root(dataset_root+"validation_images/casa_"+str(count)))
        draw_from_coco(id, result, validation_path)
        count+=1



main()