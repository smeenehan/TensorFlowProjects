from math import ceil
from network.region_proposal import RPN, ProposalLayer
import numpy as np
import tensorflow as tf

class TestRPN(tf.test.TestCase):

    def setUp(self):
        self.num_channels = 32
        self.anchors_per_loc = 4
        self.anchor_stride = 2
        self.rpn = RPN('channels_last', num_channels=self.num_channels, 
                       anchors_per_loc=self.anchors_per_loc,
                       anchor_stride=self.anchor_stride)

    def test_rpn_call(self):
        N, H, W, C = 10, 7, 7, 256
        num_anchors = ceil(H/self.anchor_stride)*ceil(W/self.anchor_stride) \
                      *self.anchors_per_loc
        features = tf.convert_to_tensor(np.random.randn(N, H, W, C).astype('float32'))    
        [logits, probs, bboxes] = self.rpn(features)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            self.assertAllEqual(tf.shape(logits).eval(), [N, num_anchors, 2])
            self.assertAllEqual(tf.shape(probs).eval(), [N, num_anchors, 2])
            self.assertAllEqual(tf.shape(bboxes).eval(), [N, num_anchors, 4])

class TestProposalLayer(tf.test.TestCase):

    def setUp(self):
        self.num_proposals = 25
        self.overlap_thresh = 0.65
        self.pl = ProposalLayer(num_proposals=self.num_proposals, 
                                overlap_thresh=self.overlap_thresh)

    def test_proposal_dims_and_range(self):
        N, small_num, large_num = 15, self.num_proposals//2, 2*self.num_proposals
        expected_dim = [N, self.num_proposals, 4]

        small_probs = tf.convert_to_tensor(
            np.random.uniform(size=(N, small_num, 2)).astype('float32'))
        large_probs = tf.convert_to_tensor(
            np.random.uniform(size=(N, large_num, 2)).astype('float32'))
        small_deltas = tf.convert_to_tensor(
            np.random.randn(N, small_num, 4).astype('float32'))
        large_deltas = tf.convert_to_tensor(
            np.random.randn(N, large_num, 4).astype('float32'))
        small_anchors = tf.convert_to_tensor(
            np.random.uniform(size=(N, small_num, 4)).astype('float32'))
        large_anchors = tf.convert_to_tensor(
            np.random.uniform(size=(N, large_num, 4)).astype('float32'))

        small_roi = self.pl([small_probs, small_deltas, small_anchors])
        large_roi = self.pl([large_probs, large_deltas, large_anchors])
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            small_roi_result = sess.run(small_roi)
            large_roi_result = sess.run(large_roi)
            self.assertAllEqual(small_roi_result.shape, expected_dim)
            self.assertTrue(np.all(small_roi_result<=np.ones_like(small_roi_result)))
            self.assertTrue(np.all(small_roi_result>=np.zeros_like(small_roi_result)))
            self.assertAllEqual(large_roi_result.shape, expected_dim)
            self.assertTrue(np.all(large_roi_result<=np.ones_like(large_roi_result)))
            self.assertTrue(np.all(large_roi_result>=np.zeros_like(large_roi_result)))

    def test_shift(self):
        probs = tf.convert_to_tensor(np.array([[[0, 1]]]).astype('float32'))
        deltas = tf.convert_to_tensor(
            np.array([[[0.05, -0.12, np.log(1.2), np.log(0.75)]]]).astype('float32'))
        init_anchors = tf.convert_to_tensor(
            np.array([[[0.1, 0.23, 0.45, 0.75]]]).astype('float32'))
        expected_anchor = np.array([0.0825, 0.2326, 0.5025, 0.6226])

        anchors = self.pl([probs, deltas, init_anchors])
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            anchors_result = sess.run(anchors)
            shifted_anchor = anchors_result[0, 0, :]
            self.assertAllClose(shifted_anchor, expected_anchor)

    def test_nms_and_pad(self):
        probs = tf.convert_to_tensor(np.array([[[0.2, 0.8], [0.3, 0.7], [0.3, 0.7]]]).astype('float32'))
        deltas = tf.convert_to_tensor(np.zeros((1, 1, 4)).astype('float32'))
        init_anchors = np.array([[[0.25, 0.25, 0.75, 0.75],
                                  [0, 0, 0.75, 0.75],
                                  [0.275, 0.275, 0.775, 0.775]]]).astype('float32')
        init_anchors_tens = tf.convert_to_tensor(init_anchors)
        expected_anchors = np.zeros((1, self.num_proposals, 4))
        expected_anchors[:, 0:2, :] = init_anchors[:, 0:2, :]

        anchors = self.pl([probs, deltas, init_anchors_tens])
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            nms_anchors = sess.run(anchors)
            self.assertAllClose(nms_anchors, expected_anchors)