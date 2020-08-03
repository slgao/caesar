import sys
import os
import pdb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ID", type=int, default=0)
parser.add_argument(
    "--type", type=str, default="RCF")
flags = parser.parse_args()
ID = flags.ID
d_type = flags.type
classes = ["RCF", "TC", "HC", "BEV", "BHC"]
classes_dict = dict(enumerate(classes))
classes_dict = {v: k for k, v in classes_dict.items()}
c_id = classes_dict.get(d_type)
label_txt = f"../data/bscan_train-class_{c_id}_{d_type}.txt"
wf = open("../data/new_" + os.path.basename(label_txt), 'w')
while(True):
    # label_txt = "../data/bscan_train-class_1_HC.txt"
    try:
        image_info = open(label_txt).readlines()[ID].split()
    except:
        sys.exit()
    
    image_path = image_info[0]
    new_bbox = []
    for bbox in image_info[1:]:
        bbox = bbox.split(",")
        bbox[0] = str(int(bbox[0]))
        bbox[1] = str(int(bbox[1]))
        bbox[2] = str(int(bbox[2]))
        bbox[3] = str(int(bbox[3]))
        # swap.
        temp = bbox[1]
        bbox[1] = bbox[3]
        bbox[3] = temp
        bbox = ','.join(bbox)
        new_bbox.append(bbox)
    print(new_bbox)
    bboxes = ' '.join(new_bbox)
    new_image_info = image_path + ' ' + bboxes
    wf.write(new_image_info + '\n')
    ID += 1
wf.close()
