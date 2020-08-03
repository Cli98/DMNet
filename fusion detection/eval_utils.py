from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import mmcv
import itertools
from terminaltables import AsciiTable
import numpy as np


def resize_bbox_to_original(bboxs, start_x, start_y):
    """
    Given bboxes from density crops, cast back coords to original images
    :param bboxs: bboxs from density crops
    :param start_x: The starting x position in original images
    :param start_y: The starting y position in original images
    :return: scaled annotations with coord matches original one.
    """
    # 4 coord for bbox: start_x, start_y, bbox_width, bbox_height
    # we only update first two column
    modify_bbox = []
    for bbox in bboxs:
        coord = bbox["bbox"]
        # coord.shape : 1*4
        coord[0] += start_x
        coord[1] += start_y
        bbox["bbox"] = coord
        modify_bbox.append(bbox)
    return modify_bbox


def wrap_initial_result(img_initial_fusion_result):
    """
    Given img_initial_fusion_result, wrap it to numpy array
    To perform class-wise nms, we need:
    1. global image id
    2. current bbox coord in global image
    3. current conf score in global image
    4. predicted category
    5. no need to record bbox id
    Leave nms to another function
    :param img_initial_fusion_result: raw annotations from initial data collecter
    :return: numpy array that is available to apply nms operation
    """
    nms_process_array = []
    for anno in img_initial_fusion_result:
        nms_process_array.append([anno[key] for key in ['image_id', 'category_id', 'score']] + anno['bbox'])
    return np.array(nms_process_array)


def class_wise_nms(current_nms_target_col, thresh, TopN):
    # NOT recommended to use. Try nms instead
    # image_id is optional
    # if final detection amount > TopN, sort by bbox and only take first TopN
    # print(current_nms_target_col.shape)
    bbox_id = np.array([i for i in range(len(current_nms_target_col))])
    truncate_result = current_nms_target_col.copy()
    current_nms_target_col[:, 0] = bbox_id
    categories = current_nms_target_col[:, 1]
    keep = []
    for category in set(categories):
        mask = current_nms_target_col[:, 1] == category
        mask = [i for i in range(len(mask)) if mask[i]]
        current_nms_target = current_nms_target_col[mask]
        current_nms_target_col = np.delete(current_nms_target_col, mask, axis=0)
        scores = current_nms_target[:, 2]
        order = scores.argsort()[::-1]
        x1 = current_nms_target[:, 3]
        y1 = current_nms_target[:, 4]
        x2 = current_nms_target[:, 5] + x1
        y2 = current_nms_target[:, 6] + y1

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        while order.size > 0:
            i = order[0]
            keep.append(int(current_nms_target[i][0]))
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        # print("Element removed: ", len(current_nms_target)-len(keep))
    if len(keep) > TopN:
        fusion_result = truncate_result[keep, :]
        scores = fusion_result[:, 2]
        keep = scores.argsort()[::-1][:TopN]
    return keep


def results2json(json_results, out_file):
    """
    Generate fused annotations to json files
    :param json_results: list, collect results to dump
    :param out_file: The output path for json file
    :return:
    """
    result_files = dict()
    result_files['bbox'] = '{}.{}.json'.format(out_file, 'bbox')
    result_files['proposal'] = '{}.{}.json'.format(out_file, 'bbox')
    mmcv.dump(json_results, result_files['bbox'])


def coco_eval(result_files,
              result_types,
              coco,
              max_dets=(100, 300, 1000),
              classwise=False):
    """
    Code from MMdetection
    Evaluate given files, with given task objective
    :param result_files: the detection file to evaluate
    :param result_types: task objective, detection? segmentation? Or something else?
    :param coco: coco object that holds all annotations
    :param max_dets: max amount of detections
    :param classwise: Conduct class-wise evaluation
    :return:
    """
    for res_type in result_types:
        assert res_type in [
            'proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'
        ]
    # borrow from mcnn
    # coco file -> cocoGt_global
    # re-load fusion result ,instead of cocoDt
    if mmcv.is_str(coco):
        coco = COCO(coco)
    assert isinstance(coco, COCO)

    if result_types == ['proposal_fast']:
        ar = fast_eval_recall(result_files, coco, np.array(max_dets))
        for i, num in enumerate(max_dets):
            print('AR@{}\t= {:.4f}'.format(num, ar[i]))
        return

    for res_type in result_types:
        if isinstance(result_files, str):
            result_file = result_files
        elif isinstance(result_files, dict):
            result_file = result_files[res_type]
        else:
            assert TypeError('result_files must be a str or dict')
        assert result_file.endswith('.json')

        coco_dets = coco.loadRes(result_file)
        img_ids = coco.getImgIds()
        iou_type = 'bbox' if res_type == 'proposal' else res_type
        cocoEval = COCOeval(coco, coco_dets, iou_type)
        cocoEval.params.imgIds = img_ids
        if res_type == 'proposal':
            cocoEval.params.useCats = 0
            cocoEval.params.maxDets = list(max_dets)
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        if classwise:
            # Compute per-category AP
            # from https://github.com/facebookresearch/detectron2/blob/03064eb5bafe4a3e5750cc7a16672daf5afe8435/detectron2/evaluation/coco_evaluation.py#L259-L283 # noqa
            precisions = cocoEval.eval['precision']
            catIds = coco.getCatIds()
            # precision has dims (iou, recall, cls, area range, max dets)
            assert len(catIds) == precisions.shape[2]

            results_per_category = []
            for idx, catId in enumerate(catIds):
                # area range index 0: all area ranges
                # max dets index -1: typically 100 per image
                nm = coco.loadCats(catId)[0]
                precision = precisions[:, :, idx, 0, -1]
                precision = precision[precision > -1]
                ap = np.mean(precision) if precision.size else float('nan')
                results_per_category.append(
                    ('{}'.format(nm['name']),
                     '{:0.3f}'.format(float(ap * 100))))

            N_COLS = min(6, len(results_per_category) * 2)
            results_flatten = list(itertools.chain(*results_per_category))
            headers = ['category', 'AP'] * (N_COLS // 2)
            results_2d = itertools.zip_longest(
                *[results_flatten[i::N_COLS] for i in range(N_COLS)])
            table_data = [headers]
            table_data += [result for result in results_2d]
            table = AsciiTable(table_data)
            print(table.table)


def nms(dets, thresh):
    '''
    Fast NMS implementation from detectron
    dets is a numpy array : num_dets, 4, but x2,y2 is height and width instead of coord
    scores ia  nump array : num_dets,
    '''
    # print("selected thr is: {}".format(thresh))
    x1 = dets[:, 3]
    y1 = dets[:, 4]
    x2 = dets[:, 5] + x1
    y2 = dets[:, 6] + y1
    scores = dets[:, 2]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]  # get boxes with more ious first

    keep = []
    while order.size > 0:
        i = order[0]  # pick maxmum iou box
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)  # maximum width
        h = np.maximum(0.0, yy2 - yy1 + 1)  # maxiumum height
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep
