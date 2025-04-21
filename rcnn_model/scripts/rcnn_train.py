import os
from detectron2.modeling import build_model
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from from_root import from_root

def main(cfg):
    os.chdir(str(from_root()))
    register_coco_instances("cubicasa_train",{},"dataset/annotations/cubicasa_train.json","dataset/")
    register_coco_instances("cubicasa_val",{},"dataset/annotations/cubicasa_test.json","dataset/")

    
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.SOLVER.IMS_PER_BATCH = 2

    model = build_model(cfg)

    model.train()
    trainer = DefaultTrainer(cfg=cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()