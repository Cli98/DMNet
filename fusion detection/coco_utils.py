import os
import json
import xml.etree.ElementTree as ET


def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.' % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.' % (name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars


def get_filename_as_int(filename):
    try:
        filename = os.path.splitext(filename)[0]
        return int(filename)
    except:
        raise NotImplementedError('Filename %s is supposed to be an integer.' % (filename))


def convert(xml_list, xml_dir, json_file, categories, affix = ".jpg"):
    START_BOUNDING_BOX_ID = 1
    json_dict = {"images": [], "type": "instances", "annotations": [],
                 "categories": []}
    bnd_id = START_BOUNDING_BOX_ID
    bbox_not_discard, total_bbox = 0, 0
    for idx, line in enumerate(xml_list):
        line = line.strip()
        print("Processing %s" % (line))
        xml_f = os.path.join(xml_dir, line)
        tree = ET.parse(xml_f)
        root = tree.getroot()
        path = get(root, 'path')
        if len(path) == 1:
            filename = os.path.basename(path[0].text)
        elif len(path) == 0:
            filename = get_and_check(root, 'filename', 1).text
        else:
            raise NotImplementedError('%d paths found in %s' % (len(path), line))
        image_id = idx
        size = get_and_check(root, 'size', 1)
        width = int(get_and_check(size, 'width', 1).text)
        height = int(get_and_check(size, 'height', 1).text)
        print("filename is: ", filename)
        if affix not in filename:
           filename += affix
        image = {'file_name': filename, 'height': height, 'width': width,
                 'id': image_id}
        json_dict['images'].append(image)
        for obj in get(root, 'object'):
            category = get_and_check(obj, 'name', 1).text
            print("The current category for bounding box is: ", category)
            if category not in categories:
                print("Category indicated not available!")
                exit(-1)
            category_id = categories[category]
            bndbox = get_and_check(obj, 'bndbox', 1)
            xmin = int(get_and_check(bndbox, 'x0', 1).text)
            ymin = int(get_and_check(bndbox, 'y0', 1).text)
            xmax = int(get_and_check(bndbox, 'x1', 1).text)
            ymax = int(get_and_check(bndbox, 'y1', 1).text)
            assert (ymax > ymin), (ymax, ymin)
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            total_bbox += 1
            if (o_height * o_width) > (height * width):
                print("Bbox out of bound, discard")
                continue
            ann = {'area': o_width * o_height, 'iscrowd': 0, 'image_id':
                image_id, 'bbox': [xmin, ymin, o_width, o_height],
                   'category_id': category_id, 'id': bnd_id, 'ignore': 0,
                   'segmentation': []}
            json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1
            bbox_not_discard += 1

    for cate, cid in categories.items():
        cat = {'supercategory': cate, 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)

    json_fp = open(json_file, 'w')
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()
