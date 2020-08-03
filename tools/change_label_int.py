import sys
import argparse

# change label 2 to 1.
parser = argparse.ArgumentParser()
parser.add_argument("--ID", type=int, default=0)
parser.add_argument(
    "--type", type=str, default="RCF")
parser.add_argument(
    "--l2c", type=int, default=0)
parser.add_argument(
    "--lc2", type=int, default=0)
flags = parser.parse_args()
ID = flags.ID
d_type = flags.type
label2change = flags.l2c
labelchange2 = flags.lc2
label_txt = f"../data/bscan_train-class_{label2change}_{d_type}.txt"
# label_txt = "../data/bscan_19020403us02.txt"
# label_txt = "../data/bscan_train-class_1_HC.txt"
wf = open(f"../data/new_bscan_train-class_{labelchange2}_{d_type}.txt", 'w')
while(True):
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
        bbox[4] = str(labelchange2)
        bbox = ','.join(bbox)
        new_bbox.append(bbox)
    print(new_bbox)
    bboxes = ' '.join(new_bbox)
    new_image_info = image_path + ' ' + bboxes
    wf.write(new_image_info + '\n')
    ID += 1
wf.close()
