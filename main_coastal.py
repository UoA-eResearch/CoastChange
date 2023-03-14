import sys
sys.path.append('/usr/local/lib/python3.8/dist-packages/')

import matplotlib
import pandas as pd
import numpy as np
import cv2

import skimage

import json

from pycocotools.coco import COCO

import os
from PIL import Image


annFile = "/home/ubuntu/d2/detectron2/coastal/coastal_coco3.json"
img_dir = "/home/ubuntu/d2/detectron2/coastal/images/"

#c:/Users/ngow210/Downloads/backup/detectron2/Code/detectron2/detectron2/model_zoo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml

#MODEL_NAME = 'COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml'
MODEL_NAME = 'COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml'
#MODEL_NAME = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
#MODEL_NAME = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
OUTPUT_DIR = "./output/"

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

class CocoTrainer(DefaultTrainer):

  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):

    if output_folder is None:
        os.makedirs("./output/", exist_ok=True)
        output_folder = "./output/coco_eval"

    return COCOEvaluator(dataset_name, cfg, False, output_folder)


def train():
  register_coco_instances("coastal_dataset", {}, annFile, img_dir)

  cfg = get_cfg()
  cfg.merge_from_file(model_zoo.get_config_file(MODEL_NAME))
  cfg.DATASETS.TRAIN = ("coastal_dataset",)
  cfg.DATASETS.TEST = ("coastal_dataset",)

  cfg.DATALOADER.NUM_WORKERS = 4
  cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL_NAME)  # Let training initialize from model zoo
  cfg.SOLVER.IMS_PER_BATCH = 4
  cfg.SOLVER.BASE_LR = 0.01 # model LR


  #cfg.SOLVER.WARMUP_ITERS = 1500
  cfg.SOLVER.MAX_ITER = 3000 #adjust up if val mAP is still rising, adjust down if overfit
  cfg.SOLVER.STEPS = (150, 250) #Milestones where the LR is reduced
  #cfg.SOLVER.GAMMA = 0.05 

  cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

  os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
  
  trainer = CocoTrainer(cfg)
  trainer.resume_or_load(resume=False)
  trainer.train()

  #MetadataCatalog.get("IVF_dataset")
  #MetadataCatalog.get("coastal_dataset").set(thing_classes=["Veg", "Coast"]) #, "bike", "truck", "bicycle"])
  #cfg.TEST.EVAL_PERIOD = 500

  cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.80
  predictor = DefaultPredictor(cfg)
  evaluator = COCOEvaluator("coastal_dataset", cfg, False, output_dir=OUTPUT_DIR)
  val_loader = build_detection_test_loader(cfg, "coastal_dataset")
  inference_on_dataset(trainer.model, val_loader, evaluator)

  #Test ***************
  cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
  cfg.DATASETS.TEST = ("coastal_dataset", )
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85   # set the testing threshold for this model
  predictor = DefaultPredictor(cfg)
  test_metadata = MetadataCatalog.get("coastal_dataset")

  for imageName in glob.glob('/home/ubuntu/d2/detectron2/coastal/test/' + '/*png'):
      im = cv2.imread(imageName)
      outputs = predictor(im)
      instances = outputs["instances"].to("cpu")
      scores = instances.scores
      classes = instances.pred_classes
      masks = instances.pred_masks

      # Extract the contour of each predicted mask and save it in a list
      contours = []
      for pred_mask in masks:
          # pred_mask is of type torch.Tensor, and the values are boolean (True, False)
          # Convert it to a 8-bit numpy array, which can then be used to find contours
          mask = pred_mask.numpy().astype('uint8')
          contour, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
          contours.append(contour[0])

      v = Visualizer(im[:, :, ::-1],
                  metadata=test_metadata, 
                  scale=1
                  )

      img_fname = imageName.split('/')
      out_img_fname = img_fname[-1]

      out = v.draw_instance_predictions(instances)
      #out = v.draw_instance_predictions(outputs["proposals"].to("cpu"))
      #cv2_imshow(out.get_image()[:, :, ::-1])
      img = out.get_image()[:, :, ::-1]
      cv2.imwrite(OUTPUT_DIR + out_img_fname, img)
      
      image_with_overlaid_predictions = im.copy()

      for contour in contours:
        img2 = cv2.drawContours(image_with_overlaid_predictions, [contour], -1, (255, 0, 0), 1)
      
      img_filename = out_img_fname.split('.')[0]

      cv2.imwrite(OUTPUT_DIR + img_filename + '_2.png', img2)

      '''
      for box in outputs["instances"].pred_boxes.to('cpu'):
        v.draw_box(box)
        v.draw_text(str(box[:2].numpy()), tuple(box[:2].numpy()))

      cv2.imwrite(OUTPUT_DIR + out_img_fname, img)
      '''
if __name__ == '__main__':
  train()