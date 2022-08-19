# Copyright (C) 2022 * Ltd. All rights reserved.
# author : Sanghyun Jo <shjo.april@gmail.com>

import xml.etree.ElementTree as ET

def read_xml(xml_path, keep_difficult=False):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    size = root.find('size')
    image_width = int(size.find('width').text)
    image_height = int(size.find('height').text)
    
    bboxes = []
    classes = []
    
    for obj in root.findall('object'):
        label = obj.find('name').text
        bbox = obj.find('bndbox')

        if not keep_difficult:
            if int(obj.find('difficult').text) == 1:
                # print(label)
                continue
        
        bbox_xmin = max(min(int(bbox.find('xmin').text.split('.')[0]), image_width - 1), 0)
        bbox_ymin = max(min(int(bbox.find('ymin').text.split('.')[0]), image_height - 1), 0)
        bbox_xmax = max(min(int(bbox.find('xmax').text.split('.')[0]), image_width - 1), 0)
        bbox_ymax = max(min(int(bbox.find('ymax').text.split('.')[0]), image_height - 1), 0)
        
        if (bbox_xmax - bbox_xmin) == 0 or (bbox_ymax - bbox_ymin) == 0:
            continue
        
        bboxes.append([bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax])
        classes.append(label)
    
    return bboxes, classes

# TODO: refine
def write_xml(xml_path, tags, image_shape):
    h, w, c = image_shape

    root = ET.Element("annotation")
    tree = ET.ElementTree(root)
    
    # for size
    size = ET.Element("size")
    
    ET.SubElement(size, "width").text = str(w)
    ET.SubElement(size, "height").text = str(h)
    ET.SubElement(size, "depth").text = str(c)

    root.append(size)

    # for object
    for tag in tags:
        object = ET.Element("object")

        ET.SubElement(object, "name").text = tag
        ET.SubElement(object, "difficult").text = '0'

        bndbox = ET.Element("bndbox")

        ET.SubElement(bndbox, "xmin").text = '0'
        ET.SubElement(bndbox, "ymin").text = '0'
        ET.SubElement(bndbox, "xmax").text = str(w-1)
        ET.SubElement(bndbox, "ymax").text = str(h-1)
        
        object.append(bndbox)

        root.append(object)

    indent(root)
    tree.write(open(xml_path, 'wb'))

def indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i