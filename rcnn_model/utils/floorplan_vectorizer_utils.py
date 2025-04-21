from PIL import Image
import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np
from pycocotools import mask
from skimage import measure
from shapely import geometry
from from_root import from_root

### Image Size Checking ###

def check_image_size_thresh(png_path, areathreshold):
        width, height = get_image_size(png_path)
        return width*height <= areathreshold

def get_image_size(png_path):
    img = Image.open(png_path)
    return img.width, img.height


### Visualization ###

def draw_from_coco(id,coco,annotated_img_dest_path,category_filter = [0,1,2,3], blank_bg = False):
    filename = coco.imgs[id]["file_name"]
    image = io.imread(str(from_root(filename)))
    if(image is not None):
        plt.figure(figsize=(image.shape[1],image.shape[0]),dpi=1)
        if(blank_bg):
            image = get_blank_image(image.shape[0],image.shape[1])
        plt.imshow(image)
        plt.axis('off')
        annotation_ids = coco.getAnnIds(imgIds=[id], catIds=category_filter, iscrowd=None)
        annotations = coco.loadAnns(annotation_ids)
        coco.showAnns(annotations)
        plt.savefig(annotated_img_dest_path,bbox_inches='tight',pad_inches=0)
        plt.clf()
        print("---")
        print("Saved Validation Image "+annotated_img_dest_path)

def get_blank_image(width,height):
    blank = io.imread(str(from_root("models/rcnn/westmoor_check/white_bg.png")))
    #return cv2.resize(blank, (0,0), fx=width, fy=height)
    return blank[:width,:height,:]


### Converting Bitmask to Polygon ###

#modifeid version of code from Waspinator on https://github.com/cocodataset/cocoapi/issues/131
def bitmask_to_polygon(id, im_id, cat_id, score, ground_truth_binary_mask, x_offset = 0, y_offset = 0, scale_factor_width = 1, scale_factor_height = 1):
    fortran_ground_truth_binary_mask = np.asfortranarray(ground_truth_binary_mask)
    encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
    ground_truth_area = mask.area(encoded_ground_truth)
    ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
    contours = measure.find_contours(ground_truth_binary_mask.numpy(), 0.5)

    bbox = []
    for i in range(0,len(ground_truth_bounding_box.tolist())):
        if(i%2 == 0):
            bbox.append(scale_factor_width*ground_truth_bounding_box.tolist()[i]+x_offset)
        else:
            bbox.append(scale_factor_height*ground_truth_bounding_box.tolist()[i]+y_offset)

    polygon = []
    for contour in contours:
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        for i in range(0,len(segmentation)):
            if(i%2 == 0):
                segmentation[i] = int(segmentation[i]*scale_factor_width)
            else:
                segmentation[i] = int(segmentation[i]*scale_factor_height)
        polygon.append(segmentation)

    segmentations = []
    if(len(polygon) >  0):
        smoothed_polygons = polygon_smoothing_and_offset(polygon[0], x_offset, y_offset)#toPolygon[0]#
        for segmentation in smoothed_polygons:
            segmentations.append(segmentation)
    
    annotation = {
            "segmentation": segmentations,
            "area": ground_truth_area.tolist(),
            "iscrowd": 0,
            "image_id": im_id,
            "bbox": bbox,
            "category_id": cat_id,
            "id": id,
            "score": score
        }
    
    return annotation

def polygon_smoothing_and_offset(polygon, x_offset, y_offset):
    points = []
    for i in range(0,int(len(polygon)/2)):
        points.append([polygon[(2*i)]+x_offset,polygon[(2*i)+1]+y_offset])
    if(len(points) < 4):
        return []
    poly = geometry.Polygon(points)
    for i in [1,2,3,5,8,12,15,18]:
        poly = poly.simplify(i)
    return [np.array(poly.exterior.coords).astype(int).ravel().tolist()[:-2]]


