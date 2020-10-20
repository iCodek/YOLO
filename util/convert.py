"""
把数据集默认的xml标注转换成YOLO数据集格式
"""
import xml.etree.ElementTree as ET
import os


name = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


def convert(size, box):
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = round(x / (size[0]), 5)
    w = round(w / (size[0]), 5)
    y = round(y / (size[1]), 5)
    h = round(h / (size[1]), 5)
    return x, y, w, h


def convert_annotation(xml, txt, classes):
    in_file = open(xml, 'r')
    out_file = open(txt, 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


def xmlConvert(txt_file, xml_path, label_path, classes=None):
    """
    :param txt_file: train.txt 所有训练图片 或 图片根目录
    :param xml_path: xml文件根目录
    :param label_path: 要生成YOLO标签的目录
    :param classes: 标签列表，自己数据集要自定义
    :return:
    """
    if classes is None:
        classes = name
    if txt_file.endswith('.txt'):
        img_name = open(txt_file).read().strip().split()
    else:
        img_name = os.listdir(txt_file)
    if not os.path.exists(label_path):
        os.makedirs(label_path)
    for image in img_name:
        convert_annotation(os.path.join(xml_path, image+'.xml'), os.path.join(label_path, image+'.txt'), classes)

