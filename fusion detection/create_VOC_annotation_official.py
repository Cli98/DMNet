import os
import scipy.misc as misc
import numpy as np
from tqdm import tqdm
from VOC_utils import save_to_xml, annotation2xml
from image_operation import clip_image_no_overlap
from file_utils import create_folders
import argparse
import cv2

"""
Code for DMnet, Loading txt annotation and transform to VOC format
The resulting images+annotations will be saved to indicated folders.
Author: Changlin Li
Code revised on : 7/17/2020

The data should be arranged in following structure before you call any function within this script:
dataset(Train/val/test)
--------images
--------Annotations (Optional, not available only when you conduct inference steps)

Sample command line to run:
python VOC2coco_official.py Folder_Name --mode train

"""

# For VisionDrone dataset only, if you work on customized dataset, please change to your category instead
# Since VOC annotation format records category in terms of their true class, we do not worry about
# the indices here.

class_list = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle",
              "bus", "motor"]
class_list = {idx: val for idx, val in enumerate(class_list)}


def format_label(mode, txt_list):
    format_data = []
    # required format: xmin, ymin, xmax, ymax, class_id, clockwise direction
    # Given format: <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,class_id
    for idx, i in enumerate(txt_list):
        coord_raw = [int(float(x)) for x in i.replace("\n", "").split(',') if len(x) != 0]
        coord = coord_raw[:6]
        if len(coord) != 6:
            # Expected to have 4 coord + 1 class, else we miss something
            print("Failed to parse annotation!")
            exit()
        if coord[2] <= 0 or coord[3] <= 0:
            print("Find out 0 height(width)!\n")
            print("This bounding box has been discarded! ")
            continue
        if not 0 < coord[-1] < 11:
            # class 0 and 11 are not in our interest, according to Visiondrone definition
            continue
        if mode != "train":
            # in this case, score is the last 2 element.
            # No consideration for score 0 in eval
            if int(coord[-2]) == 0:
                continue
            if int(coord_raw[-2]) == 2:
                continue
        bbox_left, bbox_top = coord[0], coord[1]
        bbox_right, bbox_bottom = coord[0] + coord[2], coord[1] + coord[3]
        # Scale class number back to range 0-9
        format_data.append([bbox_left, bbox_top, bbox_right, bbox_bottom, coord[-1] - 1])
    return np.array(format_data)


def parse_args():
    parser = argparse.ArgumentParser(
        description='DMNet--Create VOC annotation')
    parser.add_argument('root_dir', default=".",
                        help='the path for source data')
    parser.add_argument('--h_split', type=int, help='The number of splits across height-axis')
    parser.add_argument('--w_split', type=int, help='The number of splits across width-axis')
    parser.add_argument('--image_prefix', default="jpg", help='the path to save precomputed distance')
    parser.add_argument('--output_folder', help='The dir to save generated images and annotations')
    parser.add_argument('--mode', default="train", help='Indicate if you are working on train/val/test set')
    parser.add_argument('--keep_ori', action='store_false', help='Need to keep original image?')
    parser.add_argument('--do_crop', action='store_true', help='Need to crop image?')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print('The category of given dataset is: ', list(class_list.values()))
    mode = args.mode
    root_dir = args.root_dir
    crop_name = args.output_folder
    affix = args.image_prefix
    h_split, w_split = args.h_split, args.w_split
    keep_ori = args.keep_ori
    do_crop = args.do_crop
    if do_crop:
        assert h_split is not None and w_split is not None, "Please indicate how would you like to crop"

    raw_data = os.path.join(root_dir, mode)
    raw_images_dir = os.path.join(raw_data, 'images')
    raw_label_dir = os.path.join(raw_data, 'annotations')

    # create folder to save train/val/test/test-challenge transformation result
    create_folders(crop_name)

    save_dir = os.path.join(".", crop_name, mode)
    img_affix = affix
    images = [i for i in os.listdir(raw_images_dir) if img_affix in i]
    labels = [i for i in os.listdir(raw_label_dir) if 'txt' in i]

    print("The location of raw image: ", raw_images_dir)
    print("The location of raw annotation: ", raw_label_dir)
    print('find image', len(images))
    print('find label', len(labels))

    width = None
    height = None
    # set overlay = True only for debugging purpose
    overlay = False
    for idx, img in tqdm(enumerate(images), total=len(images)):
        img_data = cv2.imread(os.path.join(raw_images_dir, img))
        txt_data = open(os.path.join(raw_label_dir, img.replace(img_affix, 'txt')), 'r').readlines()
        box = format_label(mode, txt_data)
        if keep_ori:
            print("Working on original image!")
            if len(box) > 0:
                annotation2xml(img.strip('.' + img_affix), save_dir, img_data, box, class_list)
                status = cv2.imwrite(os.path.join(save_dir, 'images', img), img_data)
                if not status:
                    print("Failed to write images!")
                    exit()
            else:
                # However, in Inference phase we will create annotation, even if no bounding boxes detected.
                print("No box for this image!")
                continue
        if do_crop:
            print("Working on cropping image!")
            clip_image_no_overlap(img.strip('.' + img_affix), img_data, box, h_split, w_split,
                                  save_dir, class_list, overlay)
