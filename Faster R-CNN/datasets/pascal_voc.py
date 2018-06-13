from glob import glob
import numpy as np
import os
from os import path
from skimage import io
import tensorflow as tf
import xml.etree.ElementTree as et

DEFAULT_PASCAL_ROOT = path.join(path.expanduser('~'), 'Documents', 'PASCAL', 'VOC2012')
PASCAL_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
                  'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 
                  'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 
                  'train', 'tvmonitor']
SET_FILES = {'train': 'train.txt', 'val': 'val.txt', 'merged': 'trainval.txt'}

MEAN_VALUES = [123.7, 116.8, 103.9]
RESIZE_DIMS = [375, 500]
MAX_NUM_OBJ = 56

def get_image_paths(root_path=DEFAULT_PASCAL_ROOT, set_type='merged'):
    image_paths, annotation_paths = [], []
    image_name_file = path.join(root_path, 'ImageSets', 'Main', SET_FILES[set_type])
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
    class_ids, class_labels, bboxes = [], [], []
    size_info = root.find('size')
    height = int(size_info.find('height').text)
    width = int(size_info.find('width').text)

    for obj in root.iter('object'):
        obj_name = obj.find('name').text
        class_labels.append(obj_name)
        class_id = PASCAL_CLASSES.index(obj_name)
        bbox_info = obj.find('bndbox')
        y1, y2 = int(bbox_info.find('ymin').text), int(bbox_info.find('ymax').text) 
        x1, x2= int(bbox_info.find('xmin').text), int(bbox_info.find('xmax').text)
        bbox = np.array([y1/height, x1/width, y2/height, x2/width], dtype='float32')
        class_ids.append(class_id)
        bboxes.append(bbox)
    return np.array(class_ids, dtype='int32'), class_labels, np.stack(bboxes)

def make_pascal_dataset(root_path, set_type='train', batch_size=1, shuffle=None):
    data_pattern = path.join(root_path, set_type+'*.record')
    file_list = glob(data_pattern)
    dataset = tf.data.TFRecordDataset(file_list)
    if shuffle is not None:
        dataset = dataset.shuffle(shuffle)
    dataset = dataset.map(_parse_pascal_record)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(_preprocess_pascal_batch)
    return dataset

def _parse_pascal_record(example_proto):
    features = {'bboxes_y1': tf.VarLenFeature(tf.float32),
                'bboxes_x1': tf.VarLenFeature(tf.float32),
                'bboxes_y2': tf.VarLenFeature(tf.float32),
                'bboxes_x2': tf.VarLenFeature(tf.float32),
                'class_ids': tf.VarLenFeature(tf.int64),
                'class_labels': tf.VarLenFeature(tf.string),
                'encoded_jpeg': tf.VarLenFeature(tf.string)}
    parsed_features = tf.parse_single_example(example_proto, features)
    
    bboxes_y1 = parsed_features['bboxes_y1'].values
    bboxes_x1 = parsed_features['bboxes_x1'].values
    bboxes_y2 = parsed_features['bboxes_y2'].values
    bboxes_x2 = parsed_features['bboxes_x2'].values
    bboxes = tf.stack([bboxes_y1, bboxes_x1, bboxes_y2, bboxes_x2], axis=1)

    class_ids = tf.cast(parsed_features['class_ids'].values, tf.int32)

    encoded = tf.reshape(parsed_features['encoded_jpeg'].values, shape=[])
    image = tf.image.decode_jpeg(encoded)
    
    image = tf.image.resize_images(image, RESIZE_DIMS)
    padding = tf.maximum(MAX_NUM_OBJ-tf.shape(class_ids)[0],0)
    class_ids = tf.pad(class_ids, [[0, padding]])
    bboxes = tf.pad(bboxes, [[0, padding], [0, 0]])
    
    return image, class_ids, bboxes

def _preprocess_pascal_batch(images, class_ids, bboxes):
    images = images-tf.convert_to_tensor(MEAN_VALUES, 
                                         dtype=tf.float32)[None, None, None, :]
    labels = {'classes': class_ids, 'bboxes': bboxes}
    return images, labels

