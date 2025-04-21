import os
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from from_root import from_root


def write_config():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
    cfg.DATASETS.TRAIN = ("inodata_train","cubicasa_train")
    cfg.DATASETS.PROPOSAL_FILES_TRAIN = ("inodata_train")
    cfg.DATASETS.TEST = ()
    cfg.SOLVER.BASE_LR = .0005
    cfg.SOLVER.MAX_ITER = 100
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
    cfg.INPUT.MASK_FORMAT = "polygon"
    cfg.MODEL.RPN.NMS_THRESH = 0.8
    cfg.SOLVER.NUM_DECAYS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")
    cfg.SOLVER.STEPS = (50,75)
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = .7
    cfg.SOLVER.GAMMA = 0.4
    cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 14
    cfg.MODEL.ROI_MASK_HEAD.NUM_CONV = 3
    cfg.TEST.DETECTIONS_PER_IMAGE = 120
# Added this extra line
    cfg.OUTPUT_DIR = str(from_root("rcnn_model/output"))

    return cfg

os.chdir(str(from_root()))
register_coco_instances("cubicasa_train",{},"dataset/annotations/cubicasa_train.json","dataset/")
register_coco_instances("inodata_train",{},"dataset/annotations/train_sampled_data.json","dataset/")
register_coco_instances("inodata_val",{},"dataset/annotations/val_sampled_data.json","dataset/")
register_coco_instances("cubicasa_val",{},"dataset/annotations/cubicasa_test.json","dataset/")