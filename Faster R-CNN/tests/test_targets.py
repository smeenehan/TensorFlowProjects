from network.targets import RPNTargetLayer, DetectionTargetLayer
import numpy as np
import tensorflow as tf

class TestRPNTargets(tf.test.TestCase):
    def setUp(self):
        self.num_train_anchors = 12
        self.rpn = RPNTargetLayer(num_train_anchors=self.num_train_anchors)

    def test_rpn_target_dims_and_range(self):
        N, num_anchors, true_objects = 10, 8*self.num_train_anchors, 3

        anchors = tf.convert_to_tensor(
            np.random.uniform(size=(N, num_anchors, 4)).astype('float32'))
        true_boxes = tf.convert_to_tensor(
            np.random.uniform(size=(N, true_objects, 4)).astype('float32'))

        target_run = self.rpn([anchors, true_boxes])
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            [target_classes, target_deltas] = sess.run(target_run)
            self.assertAllEqual(target_classes.shape, [N, num_anchors])
            self.assertTrue(np.all(target_classes<=np.ones_like(target_classes)))
            self.assertTrue(np.all(target_classes>=-1*np.ones_like(target_classes)))
            self.assertAllEqual(target_deltas.shape, [N, num_anchors, 4])

    def test_rpn_target_values(self):
        anchors = tf.convert_to_tensor(
            np.array([[[0.21, 0.19, 0.39, 0.41],
                       [0.25, 0.1, 0.35, 0.4], 
                       [0.15, 0.15, 0.21, 0.21],
                       [0.19, 0.21, 0.41, 0.39],
                       [0.41, 0.49, 0.59, 0.61]]]).astype('float32'))
        true_boxes = tf.convert_to_tensor(
            np.array([[[0.2, 0.2, 0.4, 0.4],
                       [0.4, 0.4, 0.6, 0.6]]]).astype('float32'))

        expect_classes = np.array([[1, -1, 0, 1, 1]])

        expect_deltas = np.zeros((1, 5, 4))
        expect_deltas[0, 0, :] = [0, 0, 5*0.105361, -5*0.095310]
        expect_deltas[0, 3, :] = [0, 0, -5*0.095310, 5*0.105361]
        expect_deltas[0, 4, :] = [0, -4.16667, 5*0.105361, 5*0.510826]

        target_run = self.rpn([anchors, true_boxes])
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            [target_classes, target_deltas] = sess.run(target_run)
            self.assertAllClose(target_classes, expect_classes)
            self.assertAllClose(target_deltas, expect_deltas, atol=1e-5)

    def test_rpn_fractions(self):
        pos_anchors = np.tile([[0.21, 0.19, 0.39, 0.41]], [3*self.num_train_anchors, 1])
        neg_anchors = np.tile([[0.15, 0.15, 0.21, 0.21]], [5*self.num_train_anchors,1])
        all_anchors = np.concatenate([pos_anchors, neg_anchors])
        anchors = tf.convert_to_tensor(all_anchors[None, :].astype('float32'))
        true_boxes = tf.convert_to_tensor(
            np.array([[[0.2, 0.2, 0.4, 0.4]]]).astype('float32'))

        target_run = self.rpn([anchors, true_boxes])
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            [target_classes, target_deltas] = sess.run(target_run)
            num_pos_train = target_classes[target_classes>0].size
            num_tot_train = target_classes[target_classes>-1].size
            self.assertEqual(num_tot_train, self.num_train_anchors)
            self.assertEqual(num_pos_train, num_tot_train//2)

    def test_out_of_bounds_anchors(self):
        oob_anchors = np.array([[[-0.56, 0.2, 0.4, 0.41],
                                 [0.75, 0.1, 1.3, 0.4], 
                                 [0.15, -0.4, 0.21, 0.21],
                                 [0.19, 0.21, 0.41, 1.02]]]).astype('float32')
        good_anchors = np.array([[[0.16, 0.2, 0.4, 0.41],
                                  [0.75, 0.1, 0.93, 0.4], 
                                  [0.15, 0.14, 0.21, 0.21],
                                  [0.19, 0.21, 0.41, 0.72]]]).astype('float32')
        all_anchors = np.concatenate([oob_anchors, good_anchors], axis=1)
        anchors = tf.convert_to_tensor(all_anchors)
        true_boxes = tf.convert_to_tensor(
            np.array([[[0.2, 0.2, 0.4, 0.4]]]).astype('float32'))

        target_run = self.rpn([anchors, true_boxes])
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            [target_classes, target_deltas] = sess.run(target_run)
            self.assertAllEqual(target_classes.shape, [1, 4])
            self.assertAllEqual(target_deltas.shape, [1, 4, 4])



class TestDetectionTargets(tf.test.TestCase):
    def setUp(self):
        self.num_train_roi = 20
        self.fraction_pos_roi = 0.3
        self.num_classes = 5
        self.det = DetectionTargetLayer(
            num_train_roi=self.num_train_roi, fraction_pos_roi=self.fraction_pos_roi,
            truth_overlap_thresh=0.5)

    def test_detection_target_dims_and_range(self):
        N, num_roi, true_objects = 10, 10*self.num_train_roi, 2

        input_roi = tf.convert_to_tensor(
            np.random.uniform(size=(N, num_roi, 4)).astype('float32'))
        true_classes = np.random.randint(0, self.num_classes, size=(N, true_objects))
        true_boxes = tf.convert_to_tensor(
            np.random.uniform(size=(N, true_objects, 4)).astype('float32'))

        target_run = self.det([input_roi, true_classes, true_boxes])
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            [output_roi, target_classes, target_deltas] = sess.run(target_run)
            self.assertAllEqual(output_roi.shape, [N, self.num_train_roi, 4])
            self.assertAllEqual(target_classes.shape, [N, self.num_train_roi])

    def test_detection_target(self):
        input_roi = tf.convert_to_tensor(
            np.array([[[0.1, 0.25, 0.3, 0.55], 
                       [0.18, 0.18, 0.35, 0.41]]]).astype('float32'))
        true_classes = tf.convert_to_tensor(np.array([[3]]).astype('int32'))
        true_boxes = tf.convert_to_tensor(
            np.array([[[0.2, 0.2, 0.4, 0.4]]]).astype('float32'))

        expect_roi = np.zeros((1, self.num_train_roi, 4))
        expect_roi[0, 0, :] = [0.18, 0.18, 0.35, 0.41]
        expect_roi[0, 1, :] = [0.1, 0.25, 0.3, 0.55]

        expect_classes = -1*np.ones((1, self.num_train_roi))
        expect_classes[0, 0] = 3
        expect_classes[0, 1] = 0

        expect_deltas = np.zeros((1, self.num_train_roi, 4))
        expect_deltas[0, 0, :] = [2.05882, 0.217391, 5*0.162519, -5*0.139762]

        target_run = self.det([input_roi, true_classes, true_boxes])
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            [output_roi, target_classes, target_deltas] = sess.run(target_run)
            self.assertAllClose(output_roi, expect_roi)
            self.assertAllClose(target_classes, expect_classes)
            self.assertAllClose(target_deltas, expect_deltas, atol=1e-5)

    def test_detection_fractions(self):
        pos_roi = np.tile([[0.21, 0.19, 0.39, 0.41]], [3*self.num_train_roi, 1])
        neg_roi = np.tile([[0.15, 0.15, 0.21, 0.21]], [5*self.num_train_roi,1])
        all_roi = np.concatenate([pos_roi, neg_roi])
        input_roi = tf.convert_to_tensor(all_roi[None, :].astype('float32'))
        true_classes = tf.convert_to_tensor(np.array([[3]]).astype('int32'))
        true_boxes = tf.convert_to_tensor(
            np.array([[[0.2, 0.2, 0.4, 0.4]]]).astype('float32'))
            
        target_run = self.det([input_roi, true_classes, true_boxes])
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            [output_roi, target_classes, target_deltas] = sess.run(target_run)
            num_pos_train = target_classes[target_classes>0].size
            num_tot_train = target_classes[target_classes>-1].size
            self.assertEqual(num_tot_train, self.num_train_roi)
            self.assertAlmostEqual(num_pos_train/num_tot_train, 
                                   self.fraction_pos_roi, places=2)

    def test_no_pos_detections(self):
        input_roi = tf.convert_to_tensor(
            np.array([[[0.1, 0.2, 0.3, 0.5], 
                       [0.6, 0.7, 0.8, 0.8]]]).astype('float32'))
        true_classes = tf.convert_to_tensor(np.array([[3, 4]]).astype('int32'))
        true_boxes = tf.convert_to_tensor(
            np.array([[[0.25, 0.45, 0.35, 0.55],
                       [0.55, 0.65, 0.65, 0.75]]]).astype('float32'))

        expect_roi = np.zeros((1, self.num_train_roi, 4))
        expect_roi[0, 0, :] = [0.6, 0.7, 0.8, 0.8]
        expect_roi[0, 1, :] = [0.1, 0.2, 0.3, 0.5]

        expect_classes = -1*np.ones((1, self.num_train_roi))
        expect_classes[0, 0] = 4
        expect_classes[0, 1] = 3

        expect_deltas = np.zeros((1, self.num_train_roi, 4))
        expect_deltas[0, 0, :] = [-5, -5, -5*0.693147, 0]
        expect_deltas[0, 1, :] = [5, 5, -5*0.693147, -5*1.098612]

        target_run = self.det([input_roi, true_classes, true_boxes])
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            [output_roi, target_classes, target_deltas] = sess.run(target_run)
            sort_inds = np.flipud(np.argsort(target_classes[0, :]))
            self.assertAllClose(output_roi[:, sort_inds, :], expect_roi)
            self.assertAllClose(target_classes[:, sort_inds], expect_classes)
            self.assertAllClose(target_deltas[:, sort_inds, :], expect_deltas, atol=1e-5)