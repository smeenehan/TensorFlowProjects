from network.roi_classifier import ROIAlign, ROIHead, DetectionLayer
import numpy as np
import tensorflow as tf
from network.utils import update_bboxes

class TestROIAlign(tf.test.TestCase):
    def setUp(self):
        self.pool_shape = [2, 2]
        self.align = ROIAlign(pool_shape=self.pool_shape)

    def test_roi_align_dims(self):
        N, H, W, C = 10, 16, 25, 256
        num_roi = 25
        expected_shape = [N, num_roi]+self.pool_shape+[C]
        features = tf.convert_to_tensor(np.random.randn(N, H, W, C).astype('float32'))
        roi = tf.convert_to_tensor(np.random.uniform(size=(N, num_roi, 4)).astype('float32'))
        roi_align = self.align([features, roi])
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            self.assertAllEqual(tf.shape(roi_align).eval(), expected_shape)

    def test_roi_align_interp(self):
        feature_map = np.outer(np.arange(1, 5), np.arange(1, 5)).astype('float32')
        features = tf.convert_to_tensor(feature_map[None, :, :, None])
        roi = tf.convert_to_tensor(np.array([[[0.15, 0.25, 0.35, 0.35]]]).astype('float32'))
        expect_align = np.array([[2.5375, 2.9725], [3.5875, 4.2025]]).astype('float32')

        roi_align = self.align([features, roi])
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            roi_align_result = sess.run(roi_align)
            self.assertAllClose(np.squeeze(roi_align_result), expect_align)

class TestROIHead(tf.test.TestCase):
    def setUp(self):
        self.pool_shape = [5, 5]
        self.num_classes = 10
        self.head = ROIHead('channels_last', pool_shape=self.pool_shape, 
                            num_channels=32, num_classes=self.num_classes, 
                            hidden_dim=64)

    def test_head_dims(self):
        N, num_roi, H, W, C = 10, 5, self.pool_shape[0], self.pool_shape[1], 128 
        features = tf.convert_to_tensor(
            np.random.randn(N, num_roi, H, W, C).astype('float32'))

        [logits, probs, deltas] = self.head(features)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            self.assertAllEqual(tf.shape(logits).eval(), [N, num_roi, self.num_classes])
            self.assertAllEqual(tf.shape(probs).eval(), [N, num_roi, self.num_classes])
            self.assertAllEqual(tf.shape(deltas).eval(), [N, num_roi, self.num_classes, 4])

class TestDetectionLayer(tf.test.TestCase):

    def setUp(self):
        self.num_detect = 3
        self.num_classes = 5
        self.det = DetectionLayer(num_detect=self.num_detect, prob_thresh=0.6, 
                                  overlap_thresh=0.4)

    def test_detection_dims_and_range(self):
        N, small_num, large_num = 15, self.num_detect//2, 2*self.num_detect
        expected_dim = [N, self.num_detect, 6]

        small_roi = tf.convert_to_tensor(
            np.random.uniform(size=(N, small_num, 4)).astype('float32'))
        large_roi = tf.convert_to_tensor(
            np.random.uniform(size=(N, large_num, 4)).astype('float32'))
        small_indices = np.random.randint(0, self.num_classes, size=(N, small_num))
        small_probs = tf.one_hot(small_indices, self.num_classes, on_value=0.9)
        large_indices = np.random.randint(0, self.num_classes, size=(N, large_num))
        large_probs = tf.one_hot(large_indices, self.num_classes, on_value=0.9)
        small_deltas = tf.convert_to_tensor(
            np.random.randn(N, small_num, self.num_classes, 4).astype('float32'))
        large_deltas = tf.convert_to_tensor(
            np.random.randn(N, large_num, self.num_classes, 4).astype('float32'))

        small_det = self.det([small_roi, small_probs, small_deltas])
        large_det = self.det([large_roi, large_probs, large_deltas])
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            small_det_result = sess.run(small_det)
            large_det_result = sess.run(large_det)
            self.assertAllEqual(small_det_result.shape, expected_dim)
            self.assertTrue(np.all(small_det_result[:, :, 0:4]<=np.ones_like(small_det_result[:, :, 0:4])))
            self.assertTrue(np.all(small_det_result>=np.zeros_like(small_det_result)))
            self.assertTrue(np.all(small_det_result[:, :, 4]<=self.num_classes*np.ones_like(small_det_result[:, :, 4])))
            self.assertTrue(np.all(small_det_result[:, :, 5]<=np.ones_like(small_det_result[:, :, 5])))
            self.assertAllEqual(large_det_result.shape, expected_dim)
            self.assertTrue(np.all(large_det_result[:, :, 0:4]<=np.ones_like(large_det_result[:, :, 0:4])))
            self.assertTrue(np.all(large_det_result>=np.zeros_like(large_det_result)))
            self.assertTrue(np.all(large_det_result[:, :, 4]<=self.num_classes*np.ones_like(large_det_result[:, :, 4])))
            self.assertTrue(np.all(large_det_result[:, :, 5]<=np.ones_like(large_det_result[:, :, 5])))

    def test_background_detection(self):
        N, num_roi = 15, 2*self.num_detect
        expected = np.zeros((N, self.num_detect, 6))

        roi = tf.convert_to_tensor(
            np.random.uniform(size=(N, num_roi, 4)).astype('float32'))
        probs = np.zeros((N, num_roi, self.num_classes)).astype('float32')
        probs[:, :, 0] = 1
        probs = tf.convert_to_tensor(probs)
        deltas = tf.convert_to_tensor(
            np.random.randn(N, num_roi, self.num_classes, 4).astype('float32'))

        null_det = self.det([roi, probs, deltas])
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            null_det_result = sess.run(null_det)
            self.assertAllClose(null_det_result, expected)

    def test_nms_and_pad(self):
        roi = np.array([[[0, 0, 0.25, 0.25],
                         [0.25, 0.25, 0.75, 0.75],
                         [0.75, 0.75, 1, 1],
                         [0.275, 0.275, 0.775, 0.775]]]).astype('float32')
        roi = tf.convert_to_tensor(roi)
        probs = tf.convert_to_tensor(
            np.array([[[0.1, 0.2, 0.4, 0.15, 0.15], 
                       [0, 0.8, 0.15, 0.05, 0], 
                       [1, 0, 0, 0, 0],
                       [0, 0.7, 0, 0, 0.3]]]).astype('float32'))
        deltas = tf.convert_to_tensor(np.zeros((1, 4, 5, 4)).astype('float32'))
        expected = np.zeros((1, 3, 6))
        expected[0, 0, :] = [0.25, 0.25, 0.75, 0.75, 1, 0.8]

        pos_det = self.det([roi, probs, deltas])
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            pos_det_result = sess.run(pos_det)
            self.assertAllClose(pos_det_result, expected)