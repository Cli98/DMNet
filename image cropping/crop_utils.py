import cv2
import os
import numpy as np
from tqdm import tqdm


def crop_image_by_window(dens_array, window_size, threshold, output_dir=None):
    """
    Given stats from EDA result, crop image by pre-defined window_size and collect result to generate
    single crops for detector to use
    However, we don't do EDA within this function, as different dataset has different property.
    It is highly suggested to conduct your own EDA process before calling this function.
    :param dens_array: density map from training/inference, used to generate density crops
    :param window_size: The kernel selected to slide on images to gather crops
    :param threshold: determine if the crops are ROI, only crop when total pixel sum exceeds threshold
    :param output_dir: The output dir to save cropping image.
    :return:
    """
    # scan and gather by window stats
    if not output_dir:
        print("please provide name for output folder.")
        return
    w_h, w_w = window_size
    for file in tqdm(dens_array, total=len(dens_array)):
        img = cv2.imread(file)
        dens = np.load(file.replace("images", "dens").replace("jpg", "npy"))
        assert img.shape[:-1] == dens.shape
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
                crops = img[slide_height:slide_height + w_h, slide_width:slide_width + w_w]
                if crops.sum() >= threshold:
                    # save crop image
                    file_name = str(height) + "_" + str(width) + "_" + file.split("/")[-1]
                    cv2.imwrite(os.path.join(output_dir, file_name), crops)
    return


def low_resolution_support(img_path, result, scale=(224, 224)):
    """
    Here we take care of low resolution cropped image, instead of upsampling,
    we crop more background instead.
    Please note that currently we only upscale those that both sides are smaller than scale.
    The reason is, for one side that is larger, if we go with more pixels, they are likely
    to be background one. If they downscale afterwards, it is more likely to harm performance.
    by calling low_reso_support, it is guaranteed that
    1. We preserve respect ratio of each crop
    2. if it is not possible to resize to given resolution, then return as large as possible
    we don't take care of bbox here. Do it when we save result.
    Current we only support single scale, multi-scale is not available
    :param img_path: image_path to collect images
    :param result: images after we connect all pixels within neighbor regions
    :param scale: images with modified resolution
    :return:
    """
    img = cv2.imread(img_path)
    if img is None:
        print("Unable to locate file!")
        return
    img_height, img_width = img.shape[:2]
    ret = []
    for count, (left, top), (right, bot), pixel_area in result:
        center = ((left + right) // 2, (top + bot) // 2)
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        ori_height, ori_width = bot - top, right - left
        scale_factor = min(max_long_edge / max(ori_height, ori_width),
                           max_short_edge / min(ori_height, ori_width))
        new_height, new_width = int(ori_height * float(scale_factor) + 0.5), int(ori_width * float(scale_factor) + 0.5)
        new_left, new_right, new_top, new_bot = center[0] - new_width // 2, center[0] + new_width // 2, \
                                                center[1] - new_height // 2, center[1] + new_height // 2
        if 0 <= new_left < new_right < img_width and 0 <= new_top < new_bot < img_height:
            new_pixel_area = (new_right - new_left) * (new_bot - new_top)
            ret.append([count, (new_left, new_top), (new_right, new_bot), new_pixel_area])
            print("upsameple image from ", ori_height, ori_width)
            print("To ", new_height, new_width)
            print()
        else:
            print("Not able to resize to given resolution as not enough pixels in boundary")
            print("New bbox proposed: ", new_left, new_right, new_top, new_bot)
            print("Resize by computing boundary instead")
            # In this case, we need to change scale, while presering aspect factor
            width_bound = min(min(abs(left - img_width), left), min(abs(right - img_width), right))
            height_bound = min(min(abs(top - img_height), top), min(abs(bot - img_height), bot))
            assert min(width_bound, height_bound) < max_short_edge, "shorter edge is not in expected range!"
            edge_factor = max_long_edge / max_short_edge
            max_long_edge = min(width_bound, height_bound)
            max_short_edge = int(edge_factor * max_long_edge + 0.5)
            scale_factor = min(max_long_edge / max(ori_height, ori_width),
                               max_short_edge / min(ori_height, ori_width))
            new_height, new_width = int(ori_height * float(scale_factor) + 0.5), int(
                ori_width * float(scale_factor) + 0.5)
            new_left, new_right, new_top, new_bot = center[0] - new_width // 2, center[0] + new_width // 2, \
                                                    center[1] - new_height // 2, center[1] + new_height // 2
            new_pixel_area = (new_right - new_left) * (new_bot - new_top)
            ret.append([count, (new_left, new_top), (new_right, new_bot), new_pixel_area])
            print("upsameple image from ", ori_height, ori_width)
            print("To modified", new_height, new_width)
            print()
    return ret


def crop_large_image(img_path, result, split_threshold=(700, 700)):
    # for generated crop image, check their size. If it larger than a certain threshold, then split by
    # following methods:
    # 1. split by halve directly(benchmark)
    dens_path = img_path.replace("jpg", "npy").replace("images", "dens")
    dens_array = np.load(dens_path)
    threshold_height, threshold_width = split_threshold
    new_result = []
    new_count = 0
    for count, top_left_coord, bot_right_coord, pixel_area in result:
        (left, top), (right, bot) = top_left_coord, bot_right_coord
        height, width = bot - top, right - left
        # by default slice by height, if it falls into scale, then slice by width
        if height > threshold_height:
            theta = np.mean(dens_array[top:bot, left:right], axis=1)
            print(theta, theta.shape)
            i = np.where(theta == np.min(theta[np.nonzero(theta)]))
            split_pixel = top + int(i[0])
            # slice interval from [a,b] to [a, split_pixel], [split_pixel,b], left,right coord remains same,
            # update count and pixel area
            new_result.append([new_count, (left, top), (right, split_pixel), int((right - left) * (split_pixel - top))])
            new_count += 1
            new_result.append([new_count, (left, split_pixel), (right, bot), int((right - left) * (bot - split_pixel))])
            new_count += 1
        elif width > threshold_width:
            theta = np.mean(dens_array[top:bot, left:right], axis=0)
            print(theta, theta.shape)
            j = np.where(theta == np.min(theta[np.nonzero(theta)]))
            split_pixel = left + int(j[0])
            # slice interval from [a,b] to [a, split_pixel], [split_pixel,b], top,bot coord remains same,
            # update count and pixel area
            new_result.append([new_count, (left, top), (split_pixel, bot), int((split_pixel - left) * (bot - top))])
            new_count += 1
            new_result.append([new_count, (split_pixel, top), (right, bot), int((right - split_pixel) * (bot - top))])
            new_count += 1
        else:
            new_result.append([new_count, top_left_coord, bot_right_coord, pixel_area])
            new_count += 1
    return new_result
