import numpy as np
import os
from os import path
from skimage import io
import xml.etree.ElementTree as et

DEFAULT_PASCAL_ROOT = path.join('C:', os.sep, 'Users', 'smmeenehan', 'Documents', 
                                'PASCAL', 'VOC2012')
PASCAL_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
                  'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 
                  'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 
                  'train', 'tvmonitor']

def get_image_paths(root_path=DEFAULT_PASCAL_ROOT):
    image_paths, annotation_paths = [], []
    image_name_file = path.join(root_path, 'ImageSets', 'Main', 'trainval.txt')
    with open(image_name_file, 'r') as file:
        image_names = file.read().splitlines()
    for name in image_names:
        image_paths.append(path.join(root_path, 'JPEGImages', name+'.jpg'))
        annotation_paths.append(path.join(root_path, 'Annotations', name+'.xml'))
    return image_paths, annotation_paths

def load_image_data(image_path, annotation_path):
    image = io.imread(image_path).astype('float32')
    class_ids, bboxes = parse_annotation(annotation_path)
    return image, class_ids, bboxes

def parse_annotation(annotation_path):
    root = et.parse(annotation_path).getroot()
    class_ids, bboxes = [], []
    size_info = root.find('size')
    height = int(size_info.find('height').text)
    width = int(size_info.find('width').text)

    for obj in root.iter('object'):
        obj_name = obj.find('name').text
        class_id = PASCAL_CLASSES.index(obj_name)
        bbox_info = obj.find('bndbox')
        y1, y2 = int(bbox_info.find('ymin').text), int(bbox_info.find('ymax').text) 
        x1, x2= int(bbox_info.find('xmin').text), int(bbox_info.find('xmax').text)
        bbox = np.array([y1/height, x1/width, y2/height, x2/width], dtype='float32')
        class_ids.append(class_id)
        bboxes.append(bbox)
    return np.array(class_ids, dtype='int32'), np.stack(bboxes)



