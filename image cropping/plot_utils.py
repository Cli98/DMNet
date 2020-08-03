import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from anno_utils import format_label
from tqdm import tqdm


def overlay_image(file, window_size, threshold,
                  output_dir=None, mode="train",
                  overlay_bbox=False, show=False, save_overlay_map=False):
    # On 01/17/2020, change input type to single file_path
    """
    Serve as overlay purpose , will be able to
        1. overlay images with bbox (via overlay_bbox)
        2. overlay images with density map
    :param file: The file to overlay
    :param window_size: The kernel selected to slide on images to gather crops
    :param threshold: determine if the crops are ROI, only crop when total pixel sum exceeds threshold
    :param output_dir: The output dir to save sample image.
    :param mode: The dataset to work on (Train/val/test)
    :param overlay_bbox: Whether to overlay bbox on images
    :param show: Whether to show visualization
    :param save_overlay_map: Whether to overlay density map with images
    :return:
    """
    if save_overlay_map and not output_dir:
        print("please provide name for output folder.")
        return
    w_h, w_w = window_size
    img = cv2.imread(file)
    dens = np.load(file.replace("images", "dens").replace("jpg", "npy"))
    # Strictly speaking, only train set should have anno to draw bbox,
    # but since we have train/val/test anno, we only disable test-challenge
    if mode != "test-challenge":
        # For test-challenge dataset, we have no ground truth and thus cannot access annotations
        anno_path = file.replace("images", "annotations").replace("jpg", "txt")
        coord_list = format_label(anno_path, mode)
    overlay_map = np.zeros(dens.shape)
    # print(img.shape, dens.shape)
    assert img.shape[:-1] == dens.shape, "Shape mismatch between input image and density map!"
    img_height, img_width = img.shape[:-1]
    for height in range(0, img_height, w_h):
        if height + w_h > img_height:
            slide_height = img_height - w_h
        else:
            slide_height = height
        for width in range(0, img_width, w_w):
            if width + w_w > img_width:
                # It's next slide will across boundary, modify end condition
                slide_width = img_width - w_w
            else:
                slide_width = width
            crops = dens[slide_height:slide_height + w_h, slide_width:slide_width + w_w]
            if crops.sum() >= threshold:
                # save crop image
                overlay_map[slide_height:slide_height + w_h, slide_width:slide_width + w_w] = 255
    if overlay_bbox and mode != "test-challenge":
        for bbox in coord_list:
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
    if show:
        plt.imshow(img)
        plt.imshow(overlay_map, alpha=0.3)
        plt.axis("off")
        plt.show()
    if save_overlay_map:
        file_name = file.split("/")[-1]
        # plt.imshow(img)
        # plt.imshow(overlay_map, alpha=0.7)
        status = cv2.imwrite(os.path.join(output_dir, file_name), overlay_map)
        if not status:
            # check status to see if we successfully save images
            print("Failed to save overlay image")
        plt.savefig(os.path.join(output_dir, file_name + '.png'))
    return overlay_map


def save_overlay_image(img_array, window_size, threshold, output_dens_dir,
                       mode="train"):
    """
    For Debug/Visualization purpose only, generate a sample with cropping area indicated
    :param img_array: The input image to crop on
    :param window_size: The kernel selected to slide on images to gather crops
    :param threshold: determine if the crops are ROI, only crop when total pixel sum exceeds threshold
    :param output_dens_dir: The output dir to save density map
    :param mode: The dataset to operate on (train/val/test)
    :return:
    """
    if not os.path.exists(output_dens_dir):
        os.makedirs(output_dens_dir)
    for img_file in tqdm(img_array, total=len(img_array)):
        overlay_map = overlay_image(img_file, window_size, threshold, output_dens_dir,
                                    mode=mode,
                                    overlay_bbox=True, show=False, save_overlay_map=True)
