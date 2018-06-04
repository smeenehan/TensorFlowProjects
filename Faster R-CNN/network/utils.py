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