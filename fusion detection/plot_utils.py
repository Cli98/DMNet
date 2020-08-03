import cv2
import os


def load_whole_img_bbox(image, box):
    """
    Given image and all of its bounding boxes and categories, overlay
    those bounding boxes on images and save overlay result to root folder
    :param image: The file path for image
    :param box: The bounding boxes of image
    :return:
    """
    img = cv2.imread(image)
    for bbox in box:
        img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0))
        text = class_list[bbox[-1]]
        cv2.putText(img, text, (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                    lineType=cv2.LINE_AA)
    cv2.imwrite("./sample.jpg", img)


def overlay_func(img_pth, raw_anns, classList, truncate_threshold, exclude_region, show):
    """
    Overlay bounding boxes, category for each bounding boxes and separate by each density region
    :param img_pth: The path to input image
    :param raw_anns: annotations for given image
    :param classList: category of the dataset
    :param truncate_threshold: amount of pixels that help filter out bounding boxes that
                                close to boundary of image
    :param exclude_region: The density regions to analyze
    :param show: whether to show the overlay map
    :return: processed annotations
    """
    I = cv2.imread(img_pth)
    cp_I = I[:]
    anns = []
    for bbox in raw_anns:
        coord = bbox["bbox"]
        class_id = bbox["category_id"]
        # this is not COCO required bbox format
        if len(I.shape)==3:
            img_height, img_width = I.shape[:-1]
        elif len(I.shape)==2:
            img_height, img_width = I.shape
        else:
            print("not able to determine image shape!\n program will quit!")
            exit(0)
        bbox_left, bbox_top = coord[0], coord[1]
        bbox_right, bbox_bottom = coord[0] + coord[2], coord[1] + coord[3]
        if truncate_threshold>0:
            if not (truncate_threshold<=bbox_left<bbox_right<=img_width-truncate_threshold) or\
                    not (truncate_threshold<=bbox_top<bbox_bottom<=img_height-truncate_threshold):
                # then coord not follows into format required. So we drop this bbox
                continue

        text = classList[class_id-1]
        anns.append(bbox)
        if show:
            cp_I = cv2.rectangle(cp_I, (int(bbox_left), int(bbox_top)), (int(bbox_right), int(bbox_bottom)),
                                 (255, 0, 0), thickness=2)
            cv2.putText(cp_I, text, (int(coord[0]), int(coord[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                        lineType=cv2.LINE_AA)

    if show:
        if exclude_region:
            text = "density_crop"
            for coord in exclude_region:
                # shape assumed to be [left, top, width, height]
                bbox_left, bbox_top = coord[0], coord[1]
                bbox_right, bbox_bottom = coord[0] + coord[2], coord[1] + coord[3]
                cp_I = cv2.rectangle(cp_I, (int(bbox_left), int(bbox_top)), (int(bbox_right), int(bbox_bottom)),
                                     (0, 255, 255), thickness=2)
                cv2.putText(cp_I, text, (int(coord[0]), int(coord[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                            lineType=cv2.LINE_AA)
        cv2.imshow("overlay_image", cp_I)
        cv2.waitKey(0)
    return anns


def overlay_bbox_img(coco, img_dir, img_id, truncate_threshold, show):
    """
    Function for DEBUG purpose only
    :param coco:coco data loader that holds all annotations for input image
    :param img_dir: The dir for image sets
    :param img_id: The unique identifier for image
    :param truncate_threshold: amount of pixels that help filter out bounding boxes that
                                close to boundary of image
    :param show: whether to show the overlay map
    :return: processed annotations
    """
    classList = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle",
                 "bus", "motor"]
    img = coco.loadImgs(img_id)
    img_pth = os.path.join(img_dir, img[0]["file_name"])
    annIds = coco.getAnnIds(imgIds=img[0]['id'], catIds = [i+1 for i in range(len(classList))], iscrowd=None)
    raw_anns = coco.loadAnns(annIds)
    anns = overlay_func(img_pth, raw_anns, classList, truncate_threshold, None, show = show)
    return anns