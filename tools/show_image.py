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
import os
import pdb
import argparse
import pdb

parser = argparse.ArgumentParser()
parser.add_argument("--ID", type=int, default=0)
parser.add_argument(
    "--type", type=str, default="RCF")
label_txt_dict = {
    "RCF": "../data/bscan_train-class_0_RCF.txt",
    "TC": "../data/bscan_train-class_1_TC.txt",
    "HC": "../data/bscan_train-class_2_HC.txt",
    "BEV": "../data/bscan_train-class_3_BEV.txt",
    "BHC": "../data/bscan_train-class_4_BHC.txt"
}
flags = parser.parse_args()
ID = flags.ID
label_txt = label_txt_dict.get(flags.type)
while (True):
    # label_txt = "../data/bscan_train-class_1_HC.txt"
    try:
        image_info = open(label_txt).readlines()[ID].split()
    except IndexError:
        ID -= 1
        print("This is the last image.")
    except TypeError:
        print("Please input the right type![RCF, HC, TC, BEV, BHC]")
        sys.exit()

    image_path = image_info[0]
    image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for bbox in image_info[1:]:
        bbox = bbox.split(",")
        image = cv2.rectangle(
            image, (int(float(bbox[0])), int(float(bbox[1]))),
            (int(float(bbox[2])), int(float(bbox[3]))), (255, 0, 0), 2
        )

    # pdb.set_trace()
    # image = Image.fromarray(np.uint8(image))
    # image.show()
    cv2.imshow(f"{ID}", image)
    cv2.moveWindow(f"{ID}", 700, 200)
    k = cv2.waitKey(0)
    if k == ord('n'):
        cv2.destroyWindow(f"{ID}")
        ID += 1
    if k == ord('p'):
        cv2.destroyWindow(f"{ID}")
        if ID == 0:
            ID = 0
        else:    
            ID -= 1
    if k == ord('q'):
        cv2.destroyWindow(f"{ID}")        
        sys.exit()
    if k == ord('s'):
        if not os.path.isdir(f"{flags.T}/"):
            os.mkdir(f"{flags.T}/")
        cv2.imwrite(f"{flags.T}/{flags.T}_{os.path.basename(image_path)}", image)
        print(f"{flags.T}_{os.path.basename(image_path)} saved in {flags.T}/")
