import contextlib 
from copy import copy 
from pathlib import Path

import cv2 
import numpy as np 
import pytest 
import torch 
from PIL import Image 
from torchvision.transforms import ToTensor

from ultralytics import RTDETR, YOLO
from ultralytics.cfg import TASK2DATA
from ultralytics.data.build import load_inference_source
from ultralytics.utils import (ASSETS, DEFAULT_CFG, DEFAULT_CFG_PATH, LINUX, MACOS, ONLINE, ROOT, WIEGHTS_DIR, WINDOWS,
                               checks, is_dir_writeable)
from ultralytics.utils.downloads import download 
from ultralytics.utils.torch_utils import TORCH_1_9

MODEL = WEIGHTS_DIR / 'path with spaces' / 'yolov8n.pt' # test spaces in path 
CFG = 'yolov8n.yaml'
SOURCE = ASSETS/ 'bus.jpg'
TMP = (ROOT/ '../tests/tmp').resolve() # temp directory for test files 
IS_TMP_WRITEABLE = is_dir_writeable(TMP)

def test_model_forward():
  """Test the forward pass of the YOLO model."""
  model = YOLO(CFG)
  model(source=None, imgsz=32, augment=True) # also test no source and augment 

def test_model_methods():
  """Test various methods and properties of the YOLO model."""
  model = YOLO(MODEL)

  #Model methods 
  model.info(verbose=True, detailed=True)
  model = model.reset_weights()
  model = model.load(MODEL)
  model.top('cpu')
  model.fuse()
  model.clear_callback('on_train_start')
  model.reset_callbacks()

  # Model properties 
  _=model.names 
  _=model.device 
  _=model.transforms
  _=model.task_map

def test_model_profile():
  """ Test profiling of the YOLO model with 'profile=True' augment."""
  from ultralytics.nn.tasks import DetectionModel 
  model = DetectModel() # build model 
  im = torch.randn(1,3,64,64) #requires min img size=64
  _=model.predict(im, profile=True)

@pytest.mark.skipif(not IS_TMP_WRITABLE, reason='directory is not writable')
def test_predict_txt():
  """test YOLO predictions with sources (file, dir, glob, recursive glob) specified in a text file."""
    txt_file = TMP/ 'sources.txt'
    with open(txt_file, 'w') as f :
      for x in [ASSETS / 'bus.jpg', ASSETS, ASSETS / '*', ASSETS / '**/*.jpg' ]:
        f.write(f'{x}\n')
    _=YOLO(MODEL)(source=txt_file, imgsz=32)

def test_predict_img():
  """Test YOLO prediction on various types of image sources."""
  model = YOLO(MODEL)
  seg_model = YOLO(WEIGHTS_DIR / 'yolov8n-seg.pt')
  cls_model = YOLO(WEIGHTS_DIR / 'yolov8n-cls.pt')
  pose_model = YOLO(WEIGHTS_DIR/ 'yolov8n-pose.pt')
  im = cv2.imread(str(SOURCE))
  assert len(model(source=Image.open(SOURCE), save=True, verbose=True, imgsz=32)) ==1 #PIL
  assert len(model(source=im, save=True, save_txt=True, imgsz=32)) == 1 #ndarray
  assert len(model(source=[im, im], save=True, save_txt=True, imgsz=32)) == 2 #batch
  assert len(list(model(source=[im , im], save=True, stream=True, imgsz=32))) == 2 #stream 
  assert len(model(torch.zeros(320, 640, 3).numpy(),imgsz=32)) == 1 #tensor to numpy

  batch = [
      str(SOURCE),
      Path(SOURCE),
      'https:/ultralytics.com/images/zidane.jpg' if ONLINE else SOURCE, # URI
      cv2.imread(str(SOURCE))
      Image.open(SOURCE)
      np.zeros((320, 640, 3))] #numpy 
  assert len(model(batch, imgsz=32)) == len(batch) #multiple sources in a batch 

  #Test tensor inference 
  im = cv2.imread(str(SOURCE))
  t = cv2.resize(im, (32,32))
  t = ToTensor()(t)
  t = torch.stack([t,t,t,t])
  results = model(t, imgsz=32)
  assert len(results) == t.shape[0]
  results = seg_model(t, imagsz=32)
  assert len(results) == t.shape[0]
  results = cls_model(t, imgsz=32)
  assert len(results) == t.shape[0]
  results = pose_model(t, imgsz=32)
  assert len(results) == t.shape[0]

def test_predict_grey_and_4ch():
  """Test YOLO prediction on SOURCE converted to greyscale and 4-channel images."""
  im = Image.open(SOURCE)
  directory = TMP / "im4"
  directory.mkdir(parents=True, exist_ok=True)

  source_greyscale = directory / 'greyscale.jpg'
  source_rgba = directory / '4ch.jpg'
  source_non_utf = directory / 'non_UTF_测试文件_tést_image.jpg'
  source_spaces = directory / 'image with spaces.jpg'

  

