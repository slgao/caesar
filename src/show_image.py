#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : show_image.py
#   Author      : YunYang1994
#   Created date: 2019-07-13 09:12:53
#   Description :
#
#================================================================

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
while(True):
    # label_txt = "../data/bscan_train.txt"
    label_txt = "../data/bscan_train-class_1_HC.txt"
    try:
        image_info = open(label_txt).readlines()[ID].split()
    except:
        sys.exit()
    
    image_path = image_info[0]
    image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for bbox in image_info[1:]:
        bbox = bbox.split(",")
        image = cv2.rectangle(image,(int(float(bbox[0])),
                                     int(float(bbox[1]))),
                                    (int(float(bbox[2])),
                                     int(float(bbox[3]))), (255,0,0), 2)
    
    # pdb.set_trace()
    # image = Image.fromarray(np.uint8(image))
    # image.show()
    cv2.imshow(f"{ID}", image)
    cv2.moveWindow(f"{ID}", 700,200);
    cv2.waitKey(1000)
    cv2.destroyWindow(f"{ID}")
    ID += 1
