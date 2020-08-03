import sys
import os
from coco_utils import convert
import argparse

"""
Code for DMnet, Loading VOC annotation and transform to COCO format

Normally it should be enough to extract your annotation to VOC format, 
which is supported by various of object detection framework. However,
there does exist the needs to obtain annotation in COCO format. And 
this script can help you.

The resulting images+annotations will be saved to indicated folders
in COCO format.
Author: Changlin Li
Code revised on : 7/17/2020

The data should be arranged in following structure before you call any function within this script:
dataset(Train/val/test)
--------images
--------Annotations (XML format, Optional, not available only when you conduct inference steps)

Sample command line to run:
python create_VOC_annotation_official.py ./mcnn_0.08_train_data --h_split 2 --w_split 3 --output_folder 
FolderName --mode train 
"""

classList = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle",
             "bus", "motor"]
# By default, coco dataset starts index of categories from 1
PRE_DEFINE_CATEGORIES = {key: idx + 1 for idx, key in enumerate(classList)}


def parse_args():
    parser = argparse.ArgumentParser(
        description='DMNet--Create COCO annotation')
    parser.add_argument('root_dir', default=".",
                        help='the path for source data')
    parser.add_argument('--mode', default="train", help='Indicate if you are working on train/val/test set')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # modified to tiling
    args = parse_args()
    mode = args.mode
    crop_name = args.root_dir
    xml_dir = os.path.join(crop_name, mode, "annotations")
    xml_list = [i for i in os.listdir(xml_dir) if 'xml' in i]
    out_json_file = os.path.join(crop_name, mode, mode + ".json")
    print("The pre-defined category is: ", PRE_DEFINE_CATEGORIES)
    convert(xml_list, xml_dir, out_json_file, PRE_DEFINE_CATEGORIES)
