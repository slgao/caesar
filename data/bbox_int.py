import cv2
import numpy as np
from PIL import Image
import sys
import pdb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ID", type=int, default=0)
flags = parser.parse_args()
ID = flags.ID
wf = open("new_bscan_train.txt", 'w')
while(True):
    label_txt = "../data/bscan_train.txt"
    # label_txt = "../data/bscan_train-class_1_HC.txt"
    try:
        image_info = open(label_txt).readlines()[ID].split()
    except:
        sys.exit()
    
    image_path = image_info[0]
    new_bbox = []
    for bbox in image_info[1:]:
        bbox = bbox.split(",")
        bbox[0] = str(int(float(bbox[0])))
        bbox[1] = str(int(float(bbox[1])))
        bbox[2] = str(int(float(bbox[2])))
        bbox[3] = str(int(float(bbox[3])))
        bbox = ','.join(bbox)
        new_bbox.append(bbox)
    print(new_bbox)
    bboxes = ' '.join(new_bbox)
    new_image_info = image_path + ' ' + bboxes
    wf.write(new_image_info + '\n')
    ID += 1
wf.close()
