#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2020 * Ltd. All rights reserved.
#
#   Editor      : EMACS
#   File name   : split_annotation.py
#   Author      : slgao
#   Created date: Mo Aug 03 2020 18:04:22
#   Description : This program split the annotations of B-scan data into single
#                 txt file associated with each image.
#
# ================================================================

import sys
import os
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
        sys.exit()
    except TypeError:
        print("Please input the right type![RCF, HC, TC, BEV, BHC]")
        sys.exit()

    image_path = image_info[0]
    dir_name = os.path.dirname(image_path)
    dir_name = os.path.abspath(dir_name)
    base_name = os.path.basename(image_path)
    anno_txt = os.path.splitext(base_name)[0] + '.txt'
    anno_txt_path = os.path.join(dir_name, anno_txt)
    with open(anno_txt_path, 'w') as wf:
        wf.write(' '.join(image_info[1:]) + '\n')
    print(' '.join(image_info[1:]))
    ID += 1
