from network.utils import bbox_overlap, remove_zero_padding
import tensorflow as tf

def bbox_loss(deltas, labels, target_deltas):
    """Compute bounding-box refinement loss for anchor or ROI targets.

    Note that, as in the original Faster R-CNN paper, we normalize the loss
    of each image by the total number of anchors, not just the number of examples
    used to calculate the loss.

    Parameters
    ----------
    deltas : tensor
        Bounding-box refinements anchors/ROI, [N, num_anchor/roi, 4]
    labels : tensor
        Class labels, [N, num_anchors/roi], class == -1 indicates an irrelevant 
        anchor/ROI not used in training, class == 0 is background
    target_deltas : tensor
        Bounding-box refinement targets, [N, num_anchor/roi, 4]
    Returns
    -------
    tensor
        Mean smooth-L1 (Huber) loss function, only counting positive (foreground) 
        cases.
    """
    relevant_indices = tf.where(labels>0)
    relevant_deltas = tf.gather_nd(deltas, relevant_indices)
    relevant_targets = tf.gather_nd(target_deltas, relevant_indices)
    default_norm = tf.cast(4*tf.shape(relevant_indices)[0], tf.float32)
    correct_norm = tf.cast(tf.size(labels), tf.float32)
    loss_L1 = tf.losses.huber_loss(labels=relevant_targets, predictions=relevant_deltas)
    return default_norm*loss_L1/correct_norm

def class_loss(logits, labels):
    """Compute class loss for anchor or ROI targets.

    Parameters
    ----------
    logits : tensor
        Classifier logits for all anchors/ROI, [N, num_anchor/roi, num_classes]
    labels : tensor
        Class labels, [N, num_anchors/roi], class == -1 indicates an irrelevant 
        anchor/ROI not used in training
    Returns
    -------
    tensor
        Mean softmax cross-entropy loss for all training anchors/ROI
    """
    relevant_indices = tf.where(labels>-1)
    relevant_logits = tf.gather_nd(logits, relevant_indices)
    relevant_labels = tf.gather_nd(labels, relevant_indices)
    return tf.losses.sparse_softmax_cross_entropy(labels=relevant_labels, 
                                                  logits=relevant_logits)

def f1_score(classes, bboxes, true_classes, true_bboxes):
    """Compute F1 score (geometric mean of precision/recall) for a given set of
    detections and ground-truths.

    Note that all the input tensors may be zero-padded to allow batching.

    Parameters
    ----------
    classes : tensor
        Class IDs for all detections, [N, num_detect]
    bboxes : tensor
        Bounding boxes for all detections, [N, num_detect, (y1, x1, y2, x2)]
    true_classes : tensor
        Ground-truth class IDs, [N, num_truth]
    true_bboxes : tensor
        Ground-truth bounding boxes, [N, num_truth, (y1, x1, y2, x2)]
    Returns
    -------
    tensor
        F1 scores for each image in the batch.
    """
    f1_scores = tf.map_fn(_f1_per_image, 
        [classes, bboxes, true_classes, true_bboxes], dtype=tf.float32)
    return f1_scores

def _f1_per_image(input_data):
    classes, bboxes, true_classes, true_bboxes = input_data

    bboxes, non_zero = remove_zero_padding(bboxes)
    classes = tf.boolean_mask(classes, non_zero)
    true_bboxes, true_non_zero = remove_zero_padding(true_bboxes)
    true_classes = tf.boolean_mask(true_classes, true_non_zero)

    detect_count = tf.shape(bboxes)[0]
    truth_count = tf.shape(true_bboxes)[0]

    overlaps = bbox_overlap(bboxes, true_bboxes)
    good_boxes = overlaps > 0.5

    classes_tile = tf.tile(classes[:, None], [1, truth_count])
    true_classes_tile = tf.tile(true_classes[None, :], [detect_count, 1])
    good_classes = tf.equal(classes_tile, true_classes_tile)

    good_detections = good_boxes & good_classes

    # Only count one "good" detection per object as a true positive
    num_positives = tf.reduce_sum(tf.cast(good_detections, tf.int32), axis=0)
    at_least_one_positive = num_positives > 0
    true_positives = tf.reduce_sum(tf.cast(at_least_one_positive, tf.int32))
    return tf.cast(2*true_positives/(truth_count+detect_count), tf.float32)