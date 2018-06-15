from network.utils import bbox_overlap, compute_bbox_deltas, remove_zero_padding
from network.utils import strip_out_of_bounds_anchors
import tensorflow as tf

class RPNTargetLayer(tf.keras.Model):
    """Given a set of anchors and ground-truth bounding boxes, determine which 
    anchors should be treated as positive RPN training samples (high IoU overlap 
    with a foreground bounding box) and which should be treated as negative, along 
    with bounding-box deltas for positive samples.

    Following the Faster R-CNN paper, we use IoU > 0.7 as a threshold for positive
    samples, IoU < 0.3 for all ground-truth boxes as a threshold for negative 
    samples, and attempt to keep a 1:1 ratio of positive and negative samples.

    Parameters
    ----------
    num_train_anchors : int
        Maximum number of training samples in a single image. Defaults to 64.
    """
    def __init__(self, num_train_anchors=256):
        super().__init__()
        self.num_train_anchors = num_train_anchors

    def call(self, input_data):
        """
        Parameters
        ----------
        input_data : list of tensors
            - Anchors used for region proposal, [N, num_anchors, (y1, x1, y2, x2)]
            - Bounding box of each ground-truth object, [N, num_truth, (y1, x1, y2, x2)]
        Returns
        -------
        list of tensors
            Class indices for each anchor, and bounding-box deltas, dimensions 
            [N, num_anchors], [N, num_anchors, (y1, x1, y2, x2))], respectively. 
            For class indices, 1, 0, and -1 indicate foreground (positive), 
            background (negative) and ignored anchors, respectively. Bounding box
            deltas are only recorded for foreground anchors, with zeros elsewhere.
        """
        anchors, true_bboxes = input_data
        [target_classes, target_deltas] = tf.map_fn(
            self._get_targets, [anchors, true_bboxes], dtype=(tf.int32, tf.float32))
        return [target_classes, target_deltas]

    def _get_targets(self, input_data):
        anchors, true_bboxes = input_data
        anchors = strip_out_of_bounds_anchors(anchors)
        overlaps = bbox_overlap(anchors, true_bboxes)
        target_classes = self._get_target_classes(overlaps)
        target_classes = self._subsample_anchors(target_classes)
        target_deltas = self._get_target_deltas(anchors, true_bboxes, 
                                                overlaps, target_classes)

        return target_classes, target_deltas

    def _get_target_classes(self, overlaps):
        num_anchors = tf.shape(overlaps)[0]
        target_classes = -1*tf.ones([num_anchors], dtype=tf.int32)

        # Set background/negative bounding boxes (IoU<0.3 for all ground-truth)
        iou_max_class = tf.argmax(overlaps, axis=1, output_type=tf.int32)
        iou_max_val = tf.reduce_max(overlaps, axis=1)
        target_classes = tf.where(
            iou_max_val<0.3, tf.zeros([num_anchors], dtype=tf.int32), target_classes)

        # Ensure each ground-truth box has at least one associated anchor
        iou_max_anchor = tf.argmax(overlaps, axis=0, output_type=tf.int32)
        pos_anchors = tf.scatter_nd(iou_max_anchor[:, None], 
            tf.ones_like(iou_max_anchor), tf.shape(target_classes))
        target_classes = tf.where(pos_anchors>0, tf.ones_like(pos_anchors), 
                                  target_classes)

        # Set foreground/positive bounding boxes (IoU>0.7 for some ground-truth)
        return tf.where(iou_max_val>=0.7, tf.ones([num_anchors], dtype=tf.int32), 
                        target_classes)

    def _subsample_anchors(self, target_classes):
        # Remove random positive samples, if necessary, to no more than 50% of mini-batch 
        pos_indices = tf.cast(tf.where(target_classes>0)[:, 0], tf.int32)
        excess_pos = tf.size(pos_indices)-self.num_train_anchors//2
        random_pos = tf.random_shuffle(pos_indices)[:excess_pos]
        revert_pos = tf.scatter_nd(random_pos[:, None], -1*tf.ones_like(random_pos),
                                   tf.shape(target_classes))
        target_classes = tf.where(revert_pos<0, revert_pos, target_classes)

        # Remove random negative samples, if necessary, to reach target mini-batch size
        remaining_pos = tf.size(pos_indices)-tf.maximum(excess_pos, 0)
        desired_neg = self.num_train_anchors-remaining_pos
        neg_indices = tf.cast(tf.where((target_classes>-1) & (target_classes<1))[:, 0], 
                              tf.int32)
        excess_neg = tf.size(neg_indices)-desired_neg
        random_neg = tf.random_shuffle(neg_indices)[:excess_neg]
        revert_neg = tf.scatter_nd(random_neg[:, None], -1*tf.ones_like(random_neg),
                                   tf.shape(target_classes))
        target_classes = tf.where(revert_neg<0, revert_neg, target_classes)
        return target_classes

    def _get_target_deltas(self, anchors, true_bboxes, overlaps, target_classes):
        target_deltas = tf.zeros_like(anchors)
        pos_indices = tf.cast(tf.where(target_classes>0)[:, 0], tf.int32)
        pos_anchors = tf.gather(anchors, pos_indices)
        pos_overlap = tf.gather(overlaps, pos_indices)
        pos_truth_indices = tf.argmax(pos_overlap, axis=1)
        pos_truth_boxes = tf.gather(true_bboxes, pos_truth_indices)
        pos_deltas = compute_bbox_deltas(pos_anchors, pos_truth_boxes)
        return tf.scatter_nd(pos_indices[:, None], pos_deltas, tf.shape(anchors))   


class DetectionTargetLayer(tf.keras.Model):
    """Post-process the raw region proposals output by the ProposalLayer during
    training, to ensure a good mix of positive proposals, that substantially overlap
    with a ground-truth bounding box, and negative proposals, that don't overlap
    much with any bounding box.

    Parameters
    ----------
    num_train_roi : int
        Number of target ROI per imageduring training. Should be less than or equal
        to the total number of ROI proposals generated by ProposalLayer. Defaults
        to 64.
    fraction_pos_roi : float
        We will try to ensure that this fraction of num_train_roi are "positive"
        examples (i.e., substantially overlap a ground-truth box). Defaults to 0.5.
    truth_overlap_thresh : float
        Threshold (in terms of IoU) determining whether an ROI overlaps enough with
        a ground-truth box to count as positive. Defaults to 0.7.
    """
    def __init__(self, num_train_roi=64, fraction_pos_roi=0.25, 
                 truth_overlap_thresh=0.5):
        super().__init__()
        self.num_train_roi = num_train_roi
        self.fraction_pos_roi = fraction_pos_roi
        self.truth_overlap_thresh = truth_overlap_thresh

    def call(self, input_data):
        """
        Parameters
        ----------
        input_data : list of tensors
            - ROI proposals in normalized coordinates, [N, num_rois, (y1, x1, y2, x2)]
            - Class scores of each ground-truth box, [N, num_truth]
            - Bounding box of each ground-truth object, [N, num_truth, (y1, x1, y2, x2)]
        Returns
        -------
        list of tensors
            proposed roi, target class indices, and target bounding boxes, 
            dimensions [N, num_train_roi, (y1, x1, y2, x2))], [N, num_train_roi], 
            [N, num_train_roi, (dy, dx, log(dh), log(dw))], respectively.
        """
        proposals, true_classes, true_bboxes = input_data

        [roi, target_classes, target_deltas] = tf.map_fn(
            self._get_targets, [proposals, true_classes, true_bboxes], 
                               dtype=(tf.float32, tf.int32, tf.float32))
        return [roi, target_classes, target_deltas]

    def _get_targets(self, input_data):
        proposals, true_classes, true_bboxes = input_data

        # Remove any zero pading or zero area boxes, so we don't need to worry 
        # about divisions by zero when computing IoU, etc.
        proposals, _ = remove_zero_padding(proposals)
        true_bboxes, true_non_zero = remove_zero_padding(true_bboxes)
        true_classes = tf.boolean_mask(true_classes, true_non_zero)

        overlaps = bbox_overlap(proposals, true_bboxes)

        # Determine positive ROIs, based on ground-truth IoU overlap
        proposal_iou_max = tf.reduce_max(overlaps, axis=1)
        pos_indices = tf.where(proposal_iou_max>=self.truth_overlap_thresh)[:, 0]
        pos_indices = tf.cast(pos_indices, tf.int32)
        
        # Ensure each ground-truth box has at least one associated roi
        iou_max_roi = tf.argmax(overlaps, axis=0, output_type=tf.int32)
        pos_indices = tf.concat([pos_indices, iou_max_roi], 0)
        pos_indices, _ = tf.unique(pos_indices)

        # Determine negative ROIs, making sure that if a ROI is already positive, 
        # it cannot be negative (e.g., if an ROI is only positive b/c it is the max
        # overlap with a ground-truth box)
        pos_already = tf.scatter_nd(pos_indices[:, None], tf.ones_like(pos_indices),
                                    tf.shape(proposal_iou_max))
        neg_indices = tf.where((proposal_iou_max<1-self.truth_overlap_thresh) &
                               (pos_already<1))[:, 0]

        # Sample positive and negative ROI to hit target fractions
        pos_count = int(self.num_train_roi*self.fraction_pos_roi)
        rand_pos_indices = tf.random_shuffle(pos_indices)[:pos_count]
        pos_count = tf.shape(rand_pos_indices)[0]

        neg_count = self.num_train_roi-pos_count
        rand_neg_indices = tf.random_shuffle(neg_indices)[:neg_count]
        neg_count = tf.shape(rand_neg_indices)[0]
        pos_roi = tf.gather(proposals, rand_pos_indices)
        neg_roi = tf.gather(proposals, rand_neg_indices)

        # For positive ROI, figure out which ground-truth box we most belong to
        pos_overlap = tf.gather(overlaps, rand_pos_indices)
        pos_truth_indices = tf.argmax(pos_overlap, axis=1)
        pos_truth_boxes = tf.gather(true_bboxes, pos_truth_indices)
        pos_truth_classes = tf.gather(true_classes, pos_truth_indices)
        pos_truth_deltas = compute_bbox_deltas(pos_roi, pos_truth_boxes)

        roi = tf.concat([pos_roi, neg_roi], axis=0)
        padding = tf.maximum(self.num_train_roi-tf.shape(roi)[0], 0)
        roi = tf.pad(roi, [(0, padding), (0, 0)])
        target_classes = tf.pad(pos_truth_classes, [(0, neg_count)])
        target_classes = tf.pad(target_classes, [(0, padding)], constant_values=-1)
        target_deltas = tf.pad(pos_truth_deltas, [(0, padding+neg_count), (0, 0)])

        return roi, target_classes, target_deltas