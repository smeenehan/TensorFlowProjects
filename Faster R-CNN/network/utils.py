from math import ceil
import numpy as np
import tensorflow as tf

def generate_anchors(scales, ratios, image_shape, feature_shape, anchor_stride):
    """Generate all anchors (fixed initial bounding box guesses) for a given 
    image and feature map.

    Parameters
    ----------
    scales : list of floats
        Anchor scale, specified in number of pixels per side for a ratio of 1.
    ratios : list of floats
        Anchor aspect ratios, specified as width/height.
    image_shape : tensor shape
        Shape of the raw images (HW, not C)
    feature_shape : tensor shape
        Shape of the feature maps output by the backbone network (HW, not C)
    anchor_stride : int
        Pixel stride of anchor points on the feature map. Typically 1 or 2.
    Returns
    -------
    tensor
        [anchors, (y1, x1, y2, x2)], where anchors is the total number of anchor
        points (product of # scales, # ratios, and # feature pixels divided by
        anchor_stride^2), and coordinates are in relative units.
    """
    scales, ratios = tf.meshgrid(tf.cast(scales, tf.float32), ratios)
    scales = tf.reshape(scales, [-1])
    sqrt_ratios = tf.sqrt(tf.reshape(ratios, [-1]))
    heights = scales/sqrt_ratios
    widths = scales*sqrt_ratios

    feature_stride = tf.cast(tf.ceil(image_shape/feature_shape), tf.int32)
    shifts_y = tf.range(0, feature_shape[0], anchor_stride)*feature_stride[0]
    shifts_x = tf.range(0, feature_shape[1], anchor_stride)*feature_stride[1]
    shifts_x, shifts_y = tf.meshgrid(tf.cast(shifts_x, tf.float32), 
                                     tf.cast(shifts_y, tf.float32))

    box_widths, box_centers_x = tf.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = tf.meshgrid(heights, shifts_y)

    box_centers = tf.reshape(tf.stack([box_centers_y, box_centers_x], axis=2), 
                             [-1, 2])
    box_sizes = tf.reshape(tf.stack([box_heights, box_widths], axis=2), [-1, 2])

    boxes = tf.concat([box_centers-0.5*box_sizes, box_centers+0.5*box_sizes], axis=1)
    scale = tf.cast(tf.concat([image_shape, image_shape], axis=0), tf.float32)
    boxes = tf.divide(boxes, scale)
    return boxes

def update_bboxes(boxes, deltas):
    """Apply bounding box refinements.

    Parameters
    ----------
    boxes : tensor
        Bounding boxes to update, in normalized coordinates, 
        [num_boxes, (y1, x1, y2, x2)]
    deltas: tensor
        Refinements to apply, [num_boxes, (dy, dx, log(dh), log(dw))]
    Returns
    -------
    tensor
        Updated bounding boxes, in normalized coordinates, 
        [num_boxes, (y1, x1, y2, x2)]
    """
    height = boxes[:, 2]-boxes[:, 0]
    center_y = boxes[:, 0]+0.5*height
    width = boxes[:, 3]-boxes[:, 1]
    center_x = boxes[:, 1]+0.5*width

    center_y += deltas[:, 0]*height
    center_x += deltas[:, 1]*width
    height *= tf.exp(deltas[:, 2])
    width *= tf.exp(deltas[:, 3])

    y1 = center_y-0.5*height
    x1 = center_x-0.5*width
    y2 = y1+height
    x2 = x1+width
    return tf.stack([y1, x1, y2, x2], axis=1, name='updated_bboxes')

def update_bboxes_batch(boxes, deltas):
    """Apply bounding box refinements to a whole batch.
    """
    height = boxes[:, :, 2]-boxes[:, :, 0]
    center_y = boxes[:, :, 0]+0.5*height
    width = boxes[:, :, 3]-boxes[:, :, 1]
    center_x = boxes[:, :, 1]+0.5*width

    center_y += deltas[:, :, 0]*height
    center_x += deltas[:, :, 1]*width
    height *= tf.exp(deltas[:, :, 2])
    width *= tf.exp(deltas[:, :, 3])

    y1 = center_y-0.5*height
    x1 = center_x-0.5*width
    y2 = y1+height
    x2 = x1+width
    return tf.stack([y1, x1, y2, x2], axis=2, name='updated_bboxes')

def bbox_overlap(boxes_1, boxes_2):
    """Compute overlap (IoU metric) of two bounding boxes

    Parameters
    ----------
    boxes_1 : tensor
        First set of bounding boxes, [N_1, (y1, x1, y2, x2)]
    boxes_2 : tensor
        Second set of bounding boxes, [N_2, (y1, x1, y2, x2)]
    Returns
    -------
    tensor
        IoU for each box pair between the two sets, [N_1, N_2]
    """
    N_1, N_2 = tf.shape(boxes_1)[0], tf.shape(boxes_2)[0]
    b_1 = tf.reshape(tf.tile(boxes_1[:, None, :], [1, N_2, 1]), [-1, 4])
    b_2 = tf.tile(boxes_2, [N_1, 1])

    y1_1, x1_1, y2_1, x2_1 = tf.split(b_1, 4, axis=1)
    y1_2, x1_2, y2_2, x2_2 = tf.split(b_2, 4, axis=1)
    y1 = tf.maximum(y1_1, y1_2)
    x1 = tf.maximum(x1_1, x1_2)
    y2 = tf.minimum(y2_1, y2_2)
    x2 = tf.minimum(x2_1, x2_2)

    intersection = tf.maximum(x2-x1, 0)*tf.maximum(y2-y1, 0)
    area_1 = (y2_1-y1_1)*(x2_1-x1_1)
    area_2 = (y2_2-y1_2)*(x2_2-x1_2)
    union = area_1+area_2-intersection
    iou = intersection/union
    return tf.reshape(iou, [N_1, N_2])

def remove_zero_padding(boxes):
    non_zero_indices = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zero_indices)
    return boxes, non_zero_indices

def compute_bbox_deltas(init, target):
    """Compute delta needed to transform one set of boxes into another.

    Parameters
    ----------
    init : tensor
        First set of bounding boxes, [N_1, (y1, x1, y2, x2)]
    target : tensor
        Second set of bounding boxes, [N_2, (y1, x1, y2, x2)]
    Returns
    -------
    tensor
        Transforms between each box pair, [N, (dy, dx, log(dh), log(dw))]
    """
    init_height = init[:, 2]-init[:, 0]
    init_width = init[:, 3]-init[:, 1] 
    init_center_y = init[:, 0]+0.5*init_height 
    init_center_x = init[:, 1]+0.5*init_width

    target_height = target[:, 2]-target[:, 0]
    target_width = target[:, 3]-target[:, 1] 
    target_center_y = target[:, 0]+0.5*target_height 
    target_center_x = target[:, 1]+0.5*target_width

    dy = (target_center_y-init_center_y)/init_height
    dx = (target_center_x-init_center_x)/init_width 
    dh = tf.log(target_height/init_height)
    dw = tf.log(target_width/init_width)

    return tf.stack([dy, dx, dh, dw], axis=1)