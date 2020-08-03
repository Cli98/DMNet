import cv2
import os
from anno_utils import format_label
from plot_utils import overlay_image
from tqdm import tqdm


def split_overlay_map(grid):
    """
    Conduct eight-connected-component methods on grid to connnect all pixel within the similar region
    :param grid: desnity mask to connect
    :return: merged regions for cropping purpose
    """
    if grid is None or grid[0] is None:
        return 0
    # Assume overlap_map is a 2d feature map
    m, n = grid.shape
    visit = [[0 for _ in range(n)] for _ in range(m)]
    count, queue, result = 0, [], []
    for i in range(m):
        for j in range(n):
            if not visit[i][j]:
                if grid[i][j] == 0:
                    visit[i][j] = 1
                    continue
                queue.append([i, j])
                top, left = float("inf"), float("inf")
                bot, right = float("-inf"), float("-inf")
                while queue:
                    i_cp, j_cp = queue.pop(0)
                    top = min(i_cp, top)
                    left = min(j_cp, left)
                    bot = max(i_cp, bot)
                    right = max(j_cp, right)
                    if 0 <= i_cp < m and 0 <= j_cp < n and not visit[i_cp][j_cp]:
                        visit[i_cp][j_cp] = 1
                        if grid[i_cp][j_cp] == 255:
                            queue.append([i_cp, j_cp + 1])
                            queue.append([i_cp + 1, j_cp])
                            queue.append([i_cp, j_cp - 1])
                            queue.append([i_cp - 1, j_cp])
                count += 1
                assert top < bot and left < right, "Coordination error!"
                pixel_area = (right - left) * (bot - top)
                result.append([count, (max(0, left), max(0, top)), (min(right, n), min(bot, m)), pixel_area])
                # compute pixel area by split_coord
    return result


def gather_split_result(img_path, result, output_img_dir,
                        output_anno_dir, mode="train"):
    """
    Collect split results after we run eight-connected-components
    We need to extract coord from merging step and output the cropped images together with their annotations
    to output image/anno dir
    :param img_path: The path for image to read-in
    :param result: merging result from eight-connected-component
    :param output_img_dir: the output dir to save image
    :param output_anno_dir: the output dir to save annotations
    :param mode: The dataset to process (Train/val/test)
    :return:
    """
    # obtain output of both cropped image and cropped annotations
    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir, exist_ok=False)
    if not os.path.exists(output_anno_dir):
        os.makedirs(output_anno_dir, exist_ok=False)
    img = cv2.imread(img_path)
    if mode != "test-challenge":
        # Please note that annotation here only for evaluation purpose, nothing to do with training
        # For test-challenge dataset we have no annotations, and thus we do not read ground truth
        anno_path = img_path.replace("images", "annotations").replace("jpg", "txt")
        txt_list = format_label(anno_path, mode)
    for count, top_left_coord, bot_right_coord, pixel_area in result:
        (left, top), (right, bot) = top_left_coord, bot_right_coord
        # left, top is the offset required
        cropped_image = img[top:bot, left:right]
        cropped_image_resolution = cropped_image.shape[0] * cropped_image.shape[1]
        if cropped_image.shape[0] == 0 or cropped_image.shape[1] == 0 or cropped_image_resolution < 70 * 70:
            continue
        # we expect no images gathered with zero height/width
        assert cropped_image.shape[0] > 0 and cropped_image.shape[1] > 0, str(top) + " " + str(bot) + " " + str(
            left) + " " + str(right)
        if mode != "test-challenge":
            # If we have ground truth, generate them as txt file. This can be further used as annotations for gt generation
            with open(os.path.join(output_anno_dir, str(top) + "_" + str(left) + "_" + str(bot) + "_" + str(right) + "_"
                                                    + img_path.split(r"/")[-1].replace("jpg", "txt")),
                      'w') as filerecorder:
                for bbox_left, bbox_top, bbox_right, bbox_bottom, raw_coord in txt_list:
                    if left <= bbox_left and right >= bbox_right and top <= bbox_top and bot >= bbox_bottom:
                        raw_coord = raw_coord.split(",")
                        raw_coord[0], raw_coord[1] = str(int(raw_coord[0]) - left), str(int(raw_coord[1]) - top)
                        raw_coord = ",".join(raw_coord)
                        filerecorder.write(raw_coord)
        # If no ground truth available, then we only export images
        status = cv2.imwrite(
            os.path.join(output_img_dir, str(top) + "_" + str(left) + "_" + str(bot) + "_" + str(right) + "_"
                         + img_path.split("/")[-1]), cropped_image)
        if not status:
            # Check if images have been saved properly
            print("Failed to save image!")
            exit()
    return


def save_cropped_result(img_array, window_size, threshold, output_dens_dir,
                        output_img_dir, output_anno_dir, mode="train"):
    """
    A wrapper to conduct all necessary operation for generating density crops
    :param img_array: The input image to crop on
    :param window_size: The kernel selected to slide on images to gather crops
    :param threshold: determine if the crops are ROI, only crop when total pixel sum exceeds threshold
    :param output_dens_dir: The output dir to save density map
    :param output_img_dir: The output dir to save images
    :param output_anno_dir: The output dir to save annotations
    :param mode: The dataset to operate on (train/val/test)
    :return:
    """
    for img_file in tqdm(img_array, total=len(img_array)):
        overlay_map = overlay_image(img_file, window_size, threshold, output_dens_dir,
                                    mode=mode,
                                    overlay_bbox=False, show=False, save_overlay_map=False)
        result = split_overlay_map(overlay_map)
        gather_split_result(img_file, result, output_img_dir, output_anno_dir, mode=mode)
