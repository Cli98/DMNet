import copy, os
import numpy as np
from VOC_utils import save_to_xml
import cv2


def clip_image(file_idx, image, boxes_all, width, height, save_dir, class_list):
    """
    Crop images with overlap
    :param file_idx: image identifier
    :param image: input image to crop
    :param boxes_all: bounding boxes that belongs to this image
    :param width: The width to crop
    :param height: The height to crop
    :return:
    """
    # Format: x1, x2 ,y1 ,y2
    shape = image.shape
    if len(boxes_all) > 0:
        for start_h in range(0, shape[0], height):
            for start_w in range(0, shape[1], width):
                boxes = copy.deepcopy(boxes_all)
                box = np.zeros_like(boxes_all)
                start_h_new = start_h
                start_w_new = start_w
                if start_h + height > shape[0]:
                    start_h_new = shape[0] - height
                if start_w + width > shape[1]:
                    start_w_new = shape[1] - width
                top_left_row = max(start_h_new, 0)
                top_left_col = max(start_w_new, 0)
                bottom_right_row = min(start_h + height, shape[0])
                bottom_right_col = min(start_w + width, shape[1])
                print("current pos:", top_left_row, bottom_right_row, top_left_col, bottom_right_col)
                subImage = image[top_left_row:bottom_right_row, top_left_col: bottom_right_col]

                box[:, 0] = boxes[:, 0] - top_left_col
                box[:, 2] = boxes[:, 2] - top_left_col

                box[:, 1] = boxes[:, 1] - top_left_row
                box[:, 3] = boxes[:, 3] - top_left_row
                # Bounding box format: xmin, ymin, xmax, ymax
                # center point x = (xmin+xmax)/2
                # center point y = (ymin+ymax)/2
                center_y = 0.5 * (box[:, 1] + box[:, 3])
                center_x = 0.5 * (box[:, 0] + box[:, 2])
                # make sure 1. center x&y >0 2. center x&y < bound
                cond1 = np.intersect1d(np.where(center_y[:] >= 0)[0], np.where(center_x[:] >= 0)[0])
                cond2 = np.intersect1d(np.where(center_y[:] <= (bottom_right_row - top_left_row))[0],
                                       np.where(center_x[:] <= (bottom_right_col - top_left_col))[0])
                idx = np.intersect1d(cond1, cond2)
                if len(idx) > 0:
                    # print("bounding_box matched with image with "+str(len(idx))+" bbox!")
                    xml = os.path.join(save_dir, 'annotations',
                                       "%s_%04d_%04d.xml" % (file_idx, top_left_row, top_left_col))
                    save_to_xml(xml, "%s_%04d_%04d.jpg" % (file_idx, top_left_row, top_left_col),
                                subImage.shape[0], subImage.shape[1], box[idx, :], class_list)
                    if subImage.shape[0] > 5 and subImage.shape[1] > 5:
                        img = os.path.join(save_dir, 'images',
                                           "%s_%04d_%04d.jpg" % (file_idx, top_left_row, top_left_col))
                        cv2.imwrite(img, subImage)


def clip_image_no_overlap(file_idx, image, boxes_all, h_split, w_split, save_dir, class_list, overlay=False):
    """
    Crop images without overlap, the resulting crops may shrink, as the function will drop
    the pixels for both sides that are not divisible.
    :param file_idx: image identifier
    :param image: input image to crop
    :param boxes_all: bounding boxes that belongs to this image
    :param h_split: how many times to split along height axis
    :param w_split:  how many times to split along width axis
    :return:
    """
    # Format: x1, x2 ,y1 ,y2
    shape = image.shape
    height_step, width_step = shape[0] // h_split, shape[1] // w_split
    resolution = height_step * width_step
    if len(boxes_all) > 0:
        for start_h in range(0, shape[0], height_step):
            for start_w in range(0, shape[1], width_step):
                boxes = copy.deepcopy(boxes_all)
                box = np.zeros_like(boxes_all)
                top_left_row = max(start_h, 0)
                top_left_col = max(start_w, 0)
                bottom_right_row = min(start_h + height_step, shape[0])
                bottom_right_col = min(start_w + width_step, shape[1])
                subImage = image[top_left_row:bottom_right_row, top_left_col: bottom_right_col]
                # we do not keep data that is out of bound
                if subImage.shape[0] * subImage.shape[1] < resolution:
                    continue

                box[:, 0] = boxes[:, 0] - top_left_col
                box[:, 2] = boxes[:, 2] - top_left_col
                box[:, 1] = boxes[:, 1] - top_left_row
                box[:, 3] = boxes[:, 3] - top_left_row
                box[:, -1] = boxes[:, -1]
                # Bounding box format: xmin, ymin, xmax, ymax
                # center point x = (xmin+xmax)/2
                # center point y = (ymin+ymax)/2
                center_y = 0.5 * (box[:, 1] + box[:, 3])
                center_x = 0.5 * (box[:, 0] + box[:, 2])

                cond1 = np.intersect1d(np.where(center_y[:] >= 0)[0], np.where(center_x[:] >= 0)[0])
                cond2 = np.intersect1d(np.where(center_y[:] <= (bottom_right_row - top_left_row))[0],
                                       np.where(center_x[:] <= (bottom_right_col - top_left_col))[0])
                idx = np.intersect1d(cond1, cond2)
                if len(idx) > 0:
                    # print("bounding_box matched with image with "+str(len(idx))+" bbox!")
                    xml = os.path.join(save_dir, 'annotations',
                                       "%s_%04d_%04d.xml" % (file_idx, top_left_row, top_left_col))
                    save_to_xml(xml, "%s_%04d_%04d.jpg" % (file_idx, top_left_row, top_left_col),
                                subImage.shape[0], subImage.shape[1], box[idx, :], class_list)
                    img = os.path.join(save_dir, 'images',
                                       "%s_%04d_%04d.jpg" % (file_idx, top_left_row, top_left_col))
                    overlay_mode = overlay
                    if overlay_mode:
                        for bbox in box[idx, :]:
                            cv2.rectangle(subImage, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0))
                            text = class_list[bbox[-1]]
                            cv2.putText(subImage, text, (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (255, 255, 255),
                                        lineType=cv2.LINE_AA)
                    cv2.imwrite(img, subImage)
                else:
                    print('no img provided as no '
                          'annotations for', "%s_%04d_%04d.jpg" % (file_idx, top_left_row, top_left_col))
