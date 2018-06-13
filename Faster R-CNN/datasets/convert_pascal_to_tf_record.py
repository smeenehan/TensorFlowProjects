import argparse
import numpy as np
import pascal_voc as pvoc
import os
from os import path
import tensorflow as tf

parser = argparse.ArgumentParser()

parser.add_argument('--root-path', default=pvoc.DEFAULT_PASCAL_ROOT)
parser.add_argument('--set', default='train', choices=pvoc.SET_FILES.keys())
parser.add_argument('--out-path', default=path.join('data', 'PASCAL_TFRecords'))

def _ndarray_to_feature(array):
    dtype = array.dtype
    if dtype == np.float32 or dtype == np.float64:
        return tf.train.Feature(float_list=tf.train.FloatList(value=array.tolist()))
    elif dtype == np.int32 or dtype == np.int64:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=array.astype('int64')))
    else:
        raise ValueError('Input should be int or float, got {}'.format(dtype))

def _get_example(image_path, annotation_path):
    with tf.gfile.GFile(image_path, 'rb') as image_file:
        encoded_jpeg = image_file.read()
    class_ids, class_labels, bboxes = pvoc.parse_annotation(annotation_path)
    class_labels = [x.encode('utf8') for x in class_labels]

    example = tf.train.Example(features=tf.train.Features(feature={
        'bboxes_y1': _ndarray_to_feature(bboxes[:, 0]),
        'bboxes_x1': _ndarray_to_feature(bboxes[:, 1]),
        'bboxes_y2': _ndarray_to_feature(bboxes[:, 2]),
        'bboxes_x2': _ndarray_to_feature(bboxes[:, 3]),
        'class_ids': _ndarray_to_feature(class_ids),
        'class_labels': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=class_labels)),
        'encoded_jpeg': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[encoded_jpeg]))
        }))
    return example

def main():
    args = parser.parse_args()
    
    os.makedirs(args.out_path, exist_ok=True)
    out_file = path.join(args.out_path, args.set+'.record')
    image_paths, annotation_paths = pvoc.get_image_paths(root_path=args.root_path,
                                                         set_type=args.set)

    with tf.python_io.TFRecordWriter(out_file) as writer:
        for img_path, ann_path in zip(image_paths, annotation_paths):
            example = _get_example(img_path, ann_path)
            writer.write(example.SerializeToString())

if __name__ == '__main__':
    main()