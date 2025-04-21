import cv2
import numpy as np
import copy
from pycocotools.coco import COCO
import shapely
from shapely import geometry
import sys
import random
from datetime import datetime
from from_root import from_root
from annotation_builder import AnnotationBuilder as AnnBuild
import pylab
pylab.rcParams['figure.figsize'] = (128.0, 160.0)
from rcnn_model.utils.floorplan_vectorizer_utils import get_image_size, draw_from_coco


# sys.path.append(str(from_root("utils")))
# from floorplan_vectorizer_utils import get_image_size, draw_from_coco


### Main functionality ###

data_directory_root = str(from_root("dataset/"))+"/"
category_filter = [2]
image_sample_room_count_threshold = 4
min_sample_size = 400
max_sample_size = 800
samples_per_image = 30

def main():
    sample_from_labelme2coco_dataset("train",data_directory_root+"annotations/","sample_data/","validation_images/")
    sample_from_labelme2coco_dataset("val",data_directory_root+"annotations/","sample_data/","validation_images/")


### Core Sampling Logic ###

#sample from dataset cocofile created by labelme2coco
#dataset_name should only be "train", "val", or "dataset" based on labelme2coco's output naming conventions
def sample_from_labelme2coco_dataset(dataset_name,annotation_source_dir,sample_img_dest_dir,validation_img_dest_dir=""):
    #initialize annbuilder
    ann_builder = AnnBuild()
    ann_builder.set_info("manual annotations of Inovonics and university provided data","inovonData","NA",datetime(2019,5,24))
    ann_builder.add_license("TODO", "TODO")
    coco = COCO(annotation_source_dir+dataset_name+".json")
    print("Coco Loaded")

    #reading
    for img_id in coco.getImgIds():
        take_samples_from_image(ann_builder, img_id, coco, sample_img_dest_dir)

    #save
    ann_builder.save_file(annotation_source_dir+dataset_name+"_sampled_data.json")

    #validation images
    if(validation_img_dest_dir != ""):
        validation_coco = COCO(annotation_source_dir+dataset_name+"_sampled_data.json")
        validation_images(dataset_name,validation_coco,validation_img_dest_dir)


def take_samples_from_image(ann_builder, img_id, coco, img_dest):
    #set up image name
    source_img_filename = coco.imgs[img_id]['file_name']
    source_img = cv2.imread(data_directory_root+source_img_filename,cv2.IMREAD_COLOR)
    img_name = source_img_filename[1:-4]
    img_name = img_name[img_name.index("/"):]
    img_name = img_name[1:]

    #set up mirroring
    mirrored_imgs = [source_img, np.fliplr(source_img), np.flipud(source_img), np.flipud(np.fliplr(source_img))]
    mirror_tags = ["","_h","_v","_hv"]
    print("Processing image "+str(img_id)+": "+img_name)

    #run sampler
    for m in range(0,len(mirrored_imgs)):
        #load mirror of image
        img = mirrored_imgs[m]
        tag = mirror_tags[m]
        mirrored_anns = mirror_coco_coordinates(coco,img_id,m)

        if len(mirrored_anns) > 10:
            #collect samples
            for i in range(0,samples_per_image):
                img_dest_path = data_directory_root+img_dest+img_name+tag+"_"+str(i)+".png"
                take_sample(ann_builder, mirrored_anns, img, img_dest_path)
        else:
            img_dest_path = data_directory_root+img_dest+img_name+tag+".png"
            take_full_image(ann_builder, mirrored_anns, img, img_dest_path)


def take_sample(ann_builder, annotations, img, img_dest_path):
    #Take a random sample with at least a certain number of room bounding boxes overlapping
    sample_annotations, cropped, room_count = random_sample_selection(annotations,img)
    while(room_count < image_sample_room_count_threshold):
        sample_annotations, cropped, room_count = random_sample_selection(annotations,img)
    
    #sav the cropped image portion of the final sample
    cv2.imwrite(img_dest_path, cropped)
    print("  sample saved to "+img_dest_path)
    sample_width, sample_height = get_image_size(img_dest_path)
    sampled_img_id = ann_builder.add_image(img_dest_path, sample_width, sample_height)

    #crop annotations
    crop_area = define_crop_area(0,0,sample_width, sample_height)
    add_cropped_annotations(ann_builder, sampled_img_id, sample_annotations, crop_area)


def take_full_image(ann_builder, annotations, img, img_dest_path):
    #sav the cropped image portion of the final sample
    cv2.imwrite(img_dest_path, img)
    print("  whole image saved to "+img_dest_path)
    width, height = get_image_size(img_dest_path)
    img_id = ann_builder.add_image(img_dest_path, width, height)
    for ann in annotations:
        ann_builder.add_annotation(img_id, ann["category_id"], segmentation_to_polygon(ann["segmentation"]))


def random_sample_selection(annotations, img):
    #get bounds of original image
    init_width = len(img)
    init_height = len(img[0])

    #randomly select a rectangle
    sample_x = random.randrange(0,init_width-min_sample_size-1)
    sample_y = random.randrange(0,init_height-min_sample_size-1)
    sample_width = random.randrange(min_sample_size,min(max_sample_size, init_width-sample_x))
    sample_height = random.randrange(min_sample_size,min(max_sample_size, init_height-sample_y))

    #create cropped image and offset annotation coordinates immediately for easier data transfer
    cropped = img[sample_y:sample_y+sample_height,sample_x:sample_x+sample_width]
    sampled_annotations, room_count = offset_annotation_coordinates(annotations,sample_x,sample_y,sample_width,sample_height)

    #return values
    return sampled_annotations, cropped, room_count


### Annotation Cropping ###

def define_crop_area(x, y, width, height):
    cropped_area = geometry.Polygon([(x,y),
                                    (x+width,y),
                                    (x+width,y+height),
                                    (x,y+height)])
    return cropped_area


def add_cropped_annotations(ann_builder, img_id, annotations, cropped_area):
    for ann in annotations:
        #get intersecting area
        poly = crop_polygon(ann["segmentation"],cropped_area,ann)

        #handle convex rooms that weren't split
        if(isinstance(poly,geometry.Polygon)):
            ann_builder.add_annotation(img_id, ann["category_id"], poly)
        
        #handle concave rooms that were split
        elif(isinstance(poly,geometry.GeometryCollection) or isinstance(poly,geometry.MultiPolygon)):
            for subpoly in poly.geoms:
                if(isinstance(subpoly,geometry.Polygon)):
                    ann_builder.add_annotation(img_id, ann["category_id"], subpoly)


def crop_polygon(segmentation,crop_area,id):
    #reformat into shapely geometry Polygon
    poly = segmentation_to_polygon(segmentation)

    #check shape validity (most common error is self overlapping)
    if(not shapely.is_valid(poly)):
        print(id)
        print(shapely.is_valid_reason(poly))
        return None
    
    #check size and return
    cropped_poly = shapely.intersection(poly,crop_area)
    if(cropped_poly.area > 0):
        return cropped_poly
    else:
        return None

def segmentation_to_polygon(segmentation):
    points = np.array(segmentation[0])
    points = points.reshape(int(len(segmentation[0])/2),2)
    return geometry.Polygon([[p[0], p[1]] for p in points])


### Applying Geometry Modifications

def mirror_coco_coordinates(coco, img_id, mirroring_index):
    #instantiate copy
    original_annotations = coco.imgToAnns[img_id]
    annotations = copy.deepcopy(original_annotations)
    new_annotations = []

    #get width
    width = coco.imgs[img_id]['width']
    height = coco.imgs[img_id]['height']

    #apply mirroring
    for ann in annotations:
        if(ann["category_id"] in category_filter):
            for i in range(0,len(ann['bbox'])):
                apply_mirroring_to_coord(ann['bbox'], i, mirroring_index, width, height)
            for i in range(0,len(ann['segmentation'][0])):
                apply_mirroring_to_coord(ann['segmentation'][0], i, mirroring_index, width, height)
            new_annotations.append(ann)
    
    return new_annotations


def apply_mirroring_to_coord(array, index, mirroring_index, width, height):
    if(index%2 == 0):
        if(mirroring_index%2==1):
            array[index] = width-array[index]
    else:
        if(mirroring_index>1):
            array[index] = height-array[index]
            

def offset_annotation_coordinates(original_annotations,x_offset,y_offset,width,height):
    #instantiate copy
    annotations = copy.deepcopy(original_annotations)
    new_annotations = []
    room_count = 0

    #apply offfset
    for ann in annotations:
        if(check_bounding_box_overlap(ann['bbox'], x_offset, y_offset, width, height)):
            room_count += 1
        for i in range(0,len(ann['bbox'])):
            apply_offset_to_coord(ann['bbox'], i, x_offset, y_offset)
        for i in range(0,len(ann['segmentation'][0])):
            apply_offset_to_coord(ann['segmentation'][0], i, x_offset, y_offset)
        new_annotations.append(ann)

    return new_annotations, room_count


def apply_offset_to_coord(array, index, x_offset, y_offset):
    if(index%2 == 0):
        array[index] -= x_offset
    else:
        array[index] -= y_offset


def check_bounding_box_overlap(bbox, x_offset, y_offset, width, height):
    boundary_threshold = 25 #so a row of rooms with a one pixel sliver within the area don't get counted
    within_horizontal_bounds = max(bbox[0],bbox[2]) >= x_offset+boundary_threshold and min(bbox[0],bbox[2]) <= x_offset+width-boundary_threshold
    within_vertical_bounds = max(bbox[1],bbox[3]) >= y_offset+boundary_threshold and min(bbox[1],bbox[3]) <= y_offset+height-boundary_threshold
    return within_horizontal_bounds and within_vertical_bounds


### Validation Display ###

def validation_images(dataset_name,coco,validation_img_target_dir):
        count = 1
        for i in np.random.choice(coco.getImgIds(),8):
            draw_from_coco(i,coco,data_directory_root+validation_img_target_dir+dataset_name+"_sampling_validation_"+str(count)+".png")
            count+=1