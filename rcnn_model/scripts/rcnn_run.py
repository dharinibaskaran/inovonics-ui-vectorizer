import os
import cv2
from pycocotools.coco import COCO
from detectron2.engine import DefaultPredictor
import matplotlib.pyplot as plt
import torch
from datetime import datetime
from rcnn_config import write_config
import sys
from from_root import from_root
from rcnn_model.preprocessing.cleaning_single_image import preprocess_image
from rcnn_model.utils.floorplan_vectorizer_utils import draw_from_coco, bitmask_to_polygon
from rcnn_model.extraction.annotation_builder import AnnotationBuilder as AnnBuild



# sys.path.append(str(from_root("preprocessing")))
# from cleaning_images import preprocessing
# sys.path.append(str(from_root("utils")))
# from floorplan_vectorizer_utils import draw_from_coco, bitmask_to_polygon
# sys.path.append(str(from_root("dataset/extraction_scripts")))
# from annotation_builder import AnnotationBuilder as AnnBuild

# results_directory = str(from_root("results"))+"/"
# sample_data_directory = str(from_root("models/rcnn/sample_data"))+"/"

results_directory = "rcnn_model/results/"
sample_data_directory = "rcnn_model/sample/"

def main(cfg,img_source_path, coco_dest_filename, val_img_dest_filename):
    os.chdir(str(from_root()))

    #configure model
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.SOLVER.IMS_PER_BATCH = 1                                           
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.DEVICE = "cpu"
    predictor = DefaultPredictor(cfg)

    #run
    prediction_runner(img_source_path, results_directory+coco_dest_filename, results_directory+val_img_dest_filename, predictor)
    #prediction_runner(img_source_path, results_directory+coco_dest_filename, results_directory+val_img_dest_filename, predictor, segmented_prediction=True, scale_factor=.5)
    print("SAVED to "+results_directory+coco_dest_filename+" and "+results_directory+val_img_dest_filename)                                                  


### Main Runner ###

def prediction_runner(filename, coco_dest_path, val_img_dest_path, predictor, segmented_prediction = False, scale_factor = 0):
    #set up annotation builder
    ann_builder = instantiate_ann_build()

    #set up image
    initImg = preprocess_image(filename)
    init_width = initImg.shape[1]
    init_height = initImg.shape[0]
    img_id = ann_builder.add_image(str(from_root(filename)), init_width, init_height)
    print("cleaned")

    #resize image
    scaled_width = 800
    scaled_height = 800
    img = cv2.resize(initImg, (scaled_width,scaled_height))
    if(scale_factor > 0):
        img = cv2.resize(initImg, (0,0), fx=scale_factor, fy=scale_factor)
        scaled_width = img.shape[1]
        scaled_height = img.shape[0]
    print("resized")

    #run prediction
    if(segmented_prediction):
        run_segmented_prediction(ann_builder, predictor, img, img_id, scaled_width, scaled_height, scale_factor)
    else:
        run_prediction(ann_builder, predictor, img, img_id, init_width/scaled_width, init_height/scaled_height)

    #save the file
    ann_builder.save_file(str(from_root(coco_dest_path)))

    #visualize
    coco = COCO(str(from_root(coco_dest_path)))
    draw_from_coco(0, coco, val_img_dest_path)


def instantiate_ann_build():
    ann_builder = AnnBuild()
    ann_builder.set_info("generated annotations of Inovonics and university provided data","inovonData","NA",datetime.now())
    ann_builder.add_license("TODO", "TODO")
    return ann_builder


def prediction_outputs_to_annotations(annotations, outputs, img_id, base_ann_id, x_offset=0, y_offset=0, h_scale_factor=1, v_scale_factor=1):
    ann_id = base_ann_id
    for i in range(0,len(outputs["instances"].to(torch.device("cpu")).pred_masks)):
        mask = outputs["instances"].to(torch.device("cpu")).pred_masks[i]
        class_id = outputs["instances"].to(torch.device("cpu")).pred_classes[i].item()
        score = outputs["instances"].to(torch.device("cpu")).scores[i].item()
        annotations.append(bitmask_to_polygon(ann_id, img_id, class_id, score, mask, x_offset=x_offset, y_offset=y_offset,scale_factor_width = h_scale_factor,scale_factor_height = v_scale_factor))
        ann_id += 1
    return annotations, ann_id


### Standard Prediction ###

def run_prediction(ann_builder, predictor, img, img_id, h_scale_factor, v_scale_factor):
    outputs = predictor(img)
    print("predicted")
    annotations = []
    annotations, annId = prediction_outputs_to_annotations(annotations, outputs, img_id, 0, 0, 0, h_scale_factor, v_scale_factor)
    ann_builder.annotations = annotations
    print("annotations converted")


### Segmented Prediction ###

def run_segmented_prediction(ann_builder, predictor, img, img_id, width, height, scale_factor, segment_size = 800):
    #initialize
    count = 1
    subimg_dest_path = "models/rcnn/westmoor_check/subimgs/subimg_"
    annotations = []
    ann_id = 0

    #iterate through segments
    for xi in range(0, int(width/segment_size)+1):
        for yi in range(0, int(height/segment_size)+1):
            #calculate subimg area
            h_base, v_base, h_boundary, v_boundary = get_subimg_area(xi,yi,width,height,segment_size)

            #save subimgs
            if(h_boundary > h_base and v_boundary > v_base):
                subimg = img[v_base:v_boundary,h_base:h_boundary,:]
                save_subimg(subimg, subimg_dest_path, count, h_boundary-h_base, v_boundary-v_base)

                #get annotations
                outputs = predictor(subimg)
                annotations, ann_id = prediction_outputs_to_annotations(annotations, outputs, img_id, ann_id, h_base*1/scale_factor, h_base*1/scale_factor, 1/scale_factor, 1/scale_factor)
            count += 1
    ann_builder.annotations = annotations


def get_subimg_area(xi,yi,img_width,img_height,segment_size):
    h_base = xi*segment_size
    v_base = yi*segment_size
    h_boundary = (xi+1)*segment_size
    if(h_boundary >= img_width):
            h_boundary = img_width-1
    v_boundary = (yi+1)*segment_size
    if(v_boundary >= img_height):
            v_boundary = img_height-1
    return h_base, v_base, h_boundary, v_boundary


def save_subimg(subimg, subimg_dest_path, count, width, height):
    plt.figure(figsize=(width, height),dpi=1)
    plt.imshow(subimg)
    plt.axis('off')
    plt.savefig(str(from_root(subimg_dest_path+str(count)+".png")),bbox_inches='tight',pad_inches=0)
    plt.clf()


def default_sample(cfg, run_index = -1):
    if(run_index >= 0):
        main(cfg,sample_data_directory+"westmoor_floor_2_floorplan.png","westmoor_result_run_"+str(run_index)+".json","westmoor_result_run_"+str(run_index)+".png")
        main(cfg,sample_data_directory+"REV2 ENSURE Layout - Springs at the Waterfront - PR-2023-3661 - CN0012630 - 11.02.2023 (1)-16.png","rev2_result_run_"+str(run_index)+".json","rev2_result_run_"+str(run_index)+".png")
        main(cfg,sample_data_directory+"F1_original.png","cubicasa_result_run_"+str(run_index)+".json","cubicasa_result_run_"+str(run_index)+".png")
    else:
        main(cfg,sample_data_directory+"westmoor_floor_2_floorplan.png","westmoor_result.json","westmoor_result.png")
        main(cfg,sample_data_directory+"REV2 ENSURE Layout - Springs at the Waterfront - PR-2023-3661 - CN0012630 - 11.02.2023 (1)-16.png","rev2_result.json","rev2_result.png")
        main(cfg,sample_data_directory+"F1_original.png","cubicasa_result.json","cubicasa_result.png")

if __name__ == "__main__":
    print("Inside Main Function Call")
    cfg = write_config()
    run_index=-1
    main(cfg,sample_data_directory+"F1_original.png","cubicasa_result_run_"+str(run_index)+".json","cubicasa_result_run_"+str(run_index)+".png")
