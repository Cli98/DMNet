def format_label(anno_path, mode="train"):
    """
    Given raw label, generate cleaned labels from raw labels and transform them for further use.
    :param anno_path: The file path for annotation to process
    :param mode: The dataset to process (Train/val/test)
    :return: processed annotations
    """
    txt_list = open(anno_path, 'r').readlines()
    coord_list = []
    for idx, i in enumerate(txt_list):
        coord_raw = [int(x) for x in i.replace("\n", "").split(',') if len(x) != 0]
        coord = coord_raw[:6]
        if len(coord) != 6:
            # 4 coord + 1 class
            print("Failed to parse annotation!")
            exit()
        if coord[2] <= 0 or coord[3] <= 0:
            print("Error encountered!\nFind out 0 height(width)!")
            print("This bounding box has been discarded! ")
            continue
            # print("Pull out corrd matrix:\n")
            # print(coord)
            # exit(-1)
        if not 0 < coord[-1] < 11:
            # class 0 and 11 are not in our interest (Only for Visiondrone dataset)
            continue
        if mode == "val":
            # in this case, score is the last 2 element.
            # No consideration for score 0 in eval
            if int(coord[-2]) == 0:
                continue
            if int(coord_raw[-2]) == 2:
                continue
        bbox_left, bbox_top = coord[0], coord[1]
        bbox_right, bbox_bottom = coord[0] + coord[2], coord[1] + coord[3]
        coord_list.append([bbox_left, bbox_top, bbox_right, bbox_bottom, i])
    return coord_list
