
import sys
sys.path.append("/workspaces/tensorflow-gpu/cocoapi/PythonAPI/pycocotools")
import os
import cv2
from detectron2.data import DatasetCatalog
import detectron2.data as ddata
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer
import random
import matplotlib.pyplot as plt
import time
from rcnn_config import write_config
from from_root import from_root
from rcnn_model.utils.floorplan_vectorizer_utils import check_image_size_thresh


# sys.path.append(str(from_root("utils")))
# from floorplan_vectorizer_utils import check_image_size_thresh

results_directory = str(from_root("results"))+"/"
max_image_size = 700*500

def main(cfg,results_filename = "eval_results.txt"):
    #update config file
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.SOLVER.IMS_PER_BATCH = 1                                           
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

    #run evaluation
    results = standard_evaluation(cfg)

    #save results
    file = open(results_directory+results_filename,"w")
    file.write(str(results))
    file.close()


### Evaluation ###

def standard_evaluation(cfg):
    #load predictor
    predictor = DefaultPredictor(cfg)
    test_data_loader = ddata.build_detection_test_loader(cfg, "inodata_val")

    #save some validation images
    save_validation_images(predictor)

    #create evaluator
    evaluator = COCOEvaluator("inodata_val",tasks={"segm","bbox"},output_dir="./eval_output",distributed=False,max_dets_per_image=50,allow_cached_coco=False)
    print("EVALUATING")
    evaluator.reset()

    #load results into evaluator
    for inputs, outputs in block_prediction(test_data_loader, predictor):
        evaluator.process(inputs,outputs)
        del inputs
        del outputs
        print("|",end="")
        time.sleep(.5)
    print("")

    #run evaluator
    results = evaluator.evaluate()
    print(results)
    print("EVALUATED")
    return results


def block_prediction(loader, predictor):
    for data in loader:
        if(check_image_size_thresh(data[0]["file_name"],max_image_size)):
            image = cv2.imread(data[0]["file_name"])
            result = predictor(image)
            yield data, [result]
            del image
            del result


### Validation Images ###

def save_validation_images(predictor):
    val_img_id = 1
    for d in random.sample(DatasetCatalog.get("inodata_val"), 16):
        save_image(d,predictor,val_img_id)
        val_img_id += 1


def save_image(d, predictor, val_img_id):
    try:
        if(check_image_size_thresh(d["file_name"],max_image_size)):
            #set load image
            val_img_dest_path = "models/rcnn/validation_images/RCNN_val_image_"+str(val_img_id)+".png"
            im = cv2.imread(d["file_name"])
            outputs = predictor(im)

            #save image
            v = Visualizer(im[:,:,::-1],scale=0.5,instance_mode=ColorMode.IMAGE_BW)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            plt.imshow(out.get_image()[:,:,::-1])
            plt.axis('off')
            plt.savefig(val_img_dest_path,bbox_inches='tight',pad_inches=0)
            print("Saved validation image to "+val_img_dest_path)
            plt.clf()
    except:
        print("ERROR SAVING IMAGE")
