from network.targets import DetectionTargetLayer
import numpy as np
import tensorflow as tf

"""Subsamples proposals and generates target box refinement, class_ids,
and masks for each.
Inputs:
proposals: [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
         be zero padded if there are not enough proposals.
gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized
        coordinates.
gt_masks: [batch, height, width, MAX_GT_INSTANCES] of boolean type
Returns: Target ROIs and corresponding class IDs, bounding box shifts,
and masks.
rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized
    coordinates
target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, NUM_CLASSES,
              (dy, dx, log(dh), log(dw), class_id)]
             Class-specific bbox refinements.
target_mask: [batch, TRAIN_ROIS_PER_IMAGE, height, width)
           Masks cropped to bbox boundaries and resized to neural
           network output size.
Note: Returned arrays might be zero padded if not enough target ROIs.
"""

"""
rois, target_class_ids, target_bbox, target_mask =\
                DetectionTargetLayer(config, name="proposal_targets")([
                    target_rois, input_gt_class_ids, gt_boxes, input_gt_masks])

class_loss = KL.Lambda(lambda x: mrcnn_class_loss_graph(*x), name="mrcnn_class_loss")(
    [target_class_ids, mrcnn_class_logits, active_class_ids])
bbox_loss = KL.Lambda(lambda x: mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss")(
    [target_bbox, target_class_ids, mrcnn_bbox])
"""


class TestDetectionTargets(tf.test.TestCase):
    def setUp(self):
        self.num_train_roi = 10
        self.num_classes = 5
        self.det = DetectionTargetLayer(
            num_train_roi=self.num_train_roi, fraction_pos_roi=0.3,
            truth_overlap_thresh=0.5)

    def test_detection_target_dims(self):
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
            self.assertAllEqual(target_deltas.shape, 
                                [N, self.num_train_roi, 4])

    def test_single_target(self):
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
        expect_deltas[0, 0, :] = [0.205882,0.0217391, 0.162519, -0.139762]

        target_run = self.det([input_roi, true_classes, true_boxes])
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            [output_roi, target_classes, target_deltas] = sess.run(target_run)
            self.assertAllClose(output_roi, expect_roi)
            self.assertAllClose(target_classes, expect_classes)
            self.assertAllClose(target_deltas, expect_deltas)
            