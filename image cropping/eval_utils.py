from anno_utils import format_label


def measure_hit_rate(anno_path, result, mode="train"):
    # How many bboxs collected hits the range?
    # Serves as a trade off between max possible bbox covered and as much pieces of cropped image as possible.
    # load bbox per file
    hit = 0
    coord_list = format_label(anno_path, mode)
    for bbox_left, bbox_top, bbox_right, bbox_bottom, _ in coord_list:
        for count, (left, top), (right, bot), pixel_area in result:
            if left <= bbox_left and right >= bbox_right and top <= bbox_top and bot >= bbox_bottom:
                # it means current bbox in region
                hit += 1
                break
    return hit, len(coord_list)
