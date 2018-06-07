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

    Note that, as in the original Faster R-CNN paper, we normalize the loss
    of each image by the total number of anchors, not just the number of examples
    used to calculate the loss.

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