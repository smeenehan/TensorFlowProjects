from itertools import product
from math import ceil
import network.utils as utils
import numpy as np
import tensorflow as tf

class TestUtils(tf.test.TestCase):
    def test_generate_anchors(self):
        N, H, W, C = 10, 600, 800, 3
        FH, FW, FC = 7, 7, 2048
        y_stride, x_stride = ceil(H/FH), ceil(W/FW)
        images = tf.random_normal((N, H, W, C))
        features = tf.random_normal((N, FH, FW, FC))

        scales = [32, 64]
        ratios = [0.5, 1]
        expected_anchor_list = []
        for scale, ratio in product(scales, ratios):
            for y0, x0 in product(range(FH), range(FW)):
                height = (scale/np.sqrt(ratio))/H
                width = (scale*np.sqrt(ratio))/W
                x, y = x0*x_stride/W, y0*y_stride/H
                expected_anchor_list.append(
                    np.array([y-height/2, x-width/2, y+height/2, x+width/2]))
        expected_anchors = np.stack(expected_anchor_list)
        expected_anchors = np.sort(expected_anchors, axis=0)

        anchors = utils.generate_anchors(scales, ratios, tf.shape(images)[1:3],
                                         tf.shape(features)[1:3], 1)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            anchors_result = sess.run(anchors)
            anchors_result = np.sort(anchors_result, axis=0)
            self.assertAllClose(anchors_result, expected_anchors)

    def test_update_bboxes(self):
        init_boxes = np.tile(np.array([[0.25, 0.25, 0.75, 0.75]]), (3, 1)).astype('float32')
        init_boxes_tens = tf.convert_to_tensor(init_boxes)
        deltas = np.array([[0, 0, 0, 0], 
                           [-9, -12, 5*np.log(0.5), 5*np.log(0.6)], 
                           [14, 8, 5*np.log(1.1), 5*np.log(2.3)]]).astype('float32')
        deltas_tens = tf.convert_to_tensor(deltas)
        expected_bboxes = np.array([[0.25, 0.25, 0.75, 0.75], 
                                    [-0.075, -0.25, 0.175, 0.05], 
                                    [0.925, 0.325, 1.475, 1.475]]).astype('float32')

        bboxes = utils.update_bboxes(init_boxes_tens, deltas_tens)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            bboxes_result = sess.run(bboxes)
            self.assertAllClose(bboxes_result, expected_bboxes)

    def test_bbox_overlap(self):
        boxes_1 = tf.convert_to_tensor(
            np.array([[0, 0, 0.25, 0.25], 
                      [0.5, 0, 1, 0.4],
                      [0.4, 0.4, 0.6, 0.6],
                      [0.75, 0.75, 1, 1]]).astype('float32'))
        boxes_2 = tf.convert_to_tensor(
            np.array([[0.25, 0.25, 0.75, 0.75], 
                      [0.5, 0.5, 0.8, 0.8]]).astype('float32'))
        expected_iou = np.array([[0.0, 0.0], [0.090909, 0.0], 
                                 [0.16, 0.083333], [0.0, 0.016666]])

        iou = utils.bbox_overlap(boxes_1, boxes_2)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            iou_result = sess.run(iou)
            self.assertAllClose(iou_result, expected_iou)

    def test_compute_bbox_deltas(self):
        init = tf.convert_to_tensor(
            np.array([[0, 0.12, 0.25, 0.22], 
                      [0.5, 0, 1, 0.4]]).astype('float32'))
        target = tf.convert_to_tensor(
            np.array([[0.25, 0.25, 0.75, 0.75], 
                      [0.5, 0.5, 0.8, 0.8]]).astype('float32'))
        expected_deltas = np.array([[15, 33, 3.465736, 8.04719], 
                                    [-2, 11.25, -2.554128, -1.438410]])

        deltas = utils.compute_bbox_deltas(init, target)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            deltas_result = sess.run(deltas)
            self.assertAllClose(deltas_result, expected_deltas)
