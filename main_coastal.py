#!/usr/bin/env python
import pandas as pd
import numpy as np
import cv2

import skimage

#import json
#import matplotlib

from pycocotools.coco import COCO

import os
from PIL import Image

annFile = '/mnt/Inputs/coco.json' #'/mnt/retrolens/coco_test.json' #"/mnt/data/IVF/detectron2/Code/detectron2/coastal/Coastal_coco.json"
img_dir = '/mnt/retrolens/' #"/mnt/data/IVF/detectron2/Code/detectron2/coastal/Images/"

MODEL_NAME = 'COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml' #'COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml'
#MODEL_NAME = "COCO-Detection/rpn_R_50_C4_1x.yaml"
OUTPUT_DIR = '/mnt/output/' #"/mnt/data/IVF/detectron2/Code/detectron2/coastal/Output/"

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from detectron2.data.datasets import register_coco_instances

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

from detectron2.utils.visualizer import ColorMode
import glob

import statistics

import os
import json
import shutil
from tqdm import tqdm

# Access the registered custom COCO dataset
dataset_name = "coastal_dataset"

class CocoTrainer(DefaultTrainer):

  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):

    if output_folder is None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_folder = OUTPUT_DIR #"/mnt/data/IVF/detectron2/Code/detectron2/coastal/Output/coco_eval"

    return COCOEvaluator(dataset_name, cfg, False, output_folder)

# Register the train dataset
def register_train_dataset(name, metadata, dataset_dict):
    DatasetCatalog.register(name, lambda: dataset_dict)
    MetadataCatalog.get(name).set(**metadata)

def train(cfg):
  trainer = CocoTrainer(cfg)
  #trainer.resume_or_load(resume=True)
  trainer.train()

  #MetadataCatalog.get("IVF_dataset")
  #MetadataCatalog.get("IVF_dataset").set(thing_classes=["Cell", "Zona"]) #, "bike", "truck", "bicycle"])

  #cfg.TEST.EVAL_PERIOD = 500

  #cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
  #cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.55 #0.85
  #predictor = DefaultPredictor(cfg)
  #evaluator = COCOEvaluator(dataset_name + '_test', cfg, output_dir=OUTPUT_DIR, distributed = False, allow_cached_coco=False)
  #val_loader = build_detection_test_loader(cfg, dataset_name + '_test')
  #inference_on_dataset(trainer.model, val_loader, evaluator)
  return

def preprocess():
  # clear previous files
  for f in os.listdir(OUTPUT_DIR):
     if (f == 'model_final.pth') or f == 'Images':
        continue
     os.remove(OUTPUT_DIR + f)
  for f in os.listdir(OUTPUT_DIR + 'Images'):
    os.remove(OUTPUT_DIR + 'Images/' + f)

  DatasetCatalog.clear()
  register_coco_instances(dataset_name, {}, annFile, img_dir)
  MetadataCatalog.get(dataset_name).set(thing_classes=["sea", "land"])
  dataset_dict = DatasetCatalog.get(dataset_name)
  metadata_dataset = MetadataCatalog.get(dataset_name) #.set(thing_classes=["sea", "land"])

  # Perform train-test split
  train_ratio = 0.8  # 80% for training
  num_images = len(dataset_dict)
  num_train = int(train_ratio * num_images)

  train_dict = dataset_dict[:num_train]  # Train set
  test_dict = dataset_dict[num_train:]   # Test set

  #register_train_dataset(dataset_name + '_train', metadata_dataset, train_dict)
  DatasetCatalog.register(dataset_name + '_train', lambda: train_dict)
  dataset_dict = DatasetCatalog.get(dataset_name + '_train')

  DatasetCatalog.register(dataset_name + '_val', lambda: test_dict)
  dataset_dict = DatasetCatalog.get(dataset_name + '_val')
  MetadataCatalog.get(dataset_name + '_val').set(thing_classes=["sea", "land"])

  # DatasetCatalog.register(dataset_name + '_test', lambda: test_dict)
  # dataset_dict = DatasetCatalog.get(dataset_name + '_test')
  # MetadataCatalog.get(dataset_name + '_test').set(thing_classes=["sea", "land"])

  # # Save train_dict as a COCO-formatted JSON file
  # with open('/mnt/helper_scripts/train_coco.json', 'w') as f:
  #     json.dump(train_dict, f)  

  # # Save the split datasets
  # train_dir = '/mnt/helper_scripts/train'
  # test_dir = '/mnt/helper_scripts/test'

  # # Save train dataset
  # for item in train_dict:
  #     image_file = item["file_name"]
  #     annotation_file = item["annotations"]
  #     shutil.copy(image_file, os.path.join(train_dir, os.path.basename(image_file)))
  #     shutil.copy(annotation_file, os.path.join(train_dir, os.path.basename(annotation_file)))


  # register_coco_instances(dataset_name + '_train', {}, '/mnt/helper_scripts/train_coco.json', img_dir)

  cfg = get_cfg()
  cfg.merge_from_file(model_zoo.get_config_file(MODEL_NAME))
  cfg.OUTPUT_DIR = OUTPUT_DIR
  
  cfg.DATASETS.TRAIN = (dataset_name + '_train',)
  cfg.DATASETS.TEST = (dataset_name + '_val',)
  #cfg.DATASETS.VAL = (dataset_name + '_val',)

  cfg.DATALOADER.NUM_WORKERS = 2
  cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL_NAME)  # Let training initialize from model zoo
  cfg.SOLVER.IMS_PER_BATCH = 16
  cfg.SOLVER.BASE_LR = 0.00001

  cfg.SOLVER.WARMUP_ITERS = 10000
  cfg.SOLVER.MAX_ITER = 10000 #adjust up if val mAP is still rising, adjust down if overfit
  #cfg.SOLVER.STEPS = (150000, 100000)
  cfg.SOLVER.GAMMA = 0.05

  cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 16
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

  os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

  trainer = CocoTrainer(cfg)
  trainer.resume_or_load(resume=True)
  trainer.train()

  TEST = False
  if TEST:
    #Test ***************
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    #cfg.DATASETS.TEST = (dataset_name + '_test', )
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
    predictor = DefaultPredictor(cfg)
    test_metadata = MetadataCatalog.get(dataset_name)
    test_dict = DatasetCatalog.get(dataset_name + '_val')

    # for imageName in glob.glob('/mnt/data/IVF/detectron2/Code/detectron2/images/coastal/' + '/*jpg'):
    for filename in tqdm(test_dict[:100]):
        imageName = filename.get("file_name")
        im = cv2.imread(imageName)
        outputs = predictor(im)
        instances = outputs["instances"].to("cpu")
        scores = instances.scores
        classes = instances.pred_classes
        masks = instances.pred_masks

        v = Visualizer(im[:, :, ::-1],
                    metadata=test_metadata, 
                    scale=1
                    )

        img_fname = imageName.split('/')
        out_img_fname = img_fname[-1]
        out_img_fname = out_img_fname.split('.')[0]
        
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        #out = v.draw_instance_predictions(outputs["proposals"].to("cpu"))
        #cv2_imshow(out.get_image()[:, :, ::-1])
        img = out.get_image()[:, :, ::-1]
        cv2.imwrite(OUTPUT_DIR + out_img_fname + '.jpg', img)

        for box in outputs["instances"].pred_boxes.to('cpu'):
          v.draw_box(box)
          #v.draw_text(str(box[:2].numpy()), tuple(box[:2].numpy()))

        cv2.imwrite(OUTPUT_DIR + 'Images/' + out_img_fname + '.jpg', img)

if __name__ == '__main__':
  cfg = preprocess()
  train(cfg)
  #test(cfg)