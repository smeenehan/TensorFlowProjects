import tensorflow as tf
from utils import update_bboxes

class RPN(tf.keras.Model):
    """Implement a Region Proposal Network.

    Takes a set of feature maps (penultimate output of a classifier network), 
    and predicts the class probability (foreground or background) and bounding 
    box refinement for a variable number of anchors at each feature map pixel.

    Parameters
    ----------
    data_format : string
        'channels_first' or 'channels_last', indicating the ordering of feature
        maps and channels.
    num_channels : int
        Number of channels to downsample input feature maps. Defaults to 512.
    anchors_per_loc : int
        Number of anchor points per feature map pixel. Defaults to 9.
    anchor_stride : int
        Pixel stride of anchor points on the feature map. Typically 1 or 2. Defaults
        to 1.
    regularizer : function
        Regularizer function applied to all weights in the network. Defaults to
        None.
    """
    def __init__(self, data_format, num_channels=512, anchors_per_loc=9, 
                 anchor_stride=1, regularizer=None):
        super().__init__()
        self.shared = tf.keras.layers.Conv2D(
            num_channels, 3, strides=anchor_stride, data_format=data_format, 
            use_bias=False, padding='same', name='shared', 
            kernel_regularizer=regularizer)

        bn_axis = 1 if data_format is 'channels_first' else 3
        self.bn = tf.layers.BatchNormalization(axis=bn_axis, name='bn')

        self.anchor_logits = tf.keras.layers.Conv2D(
            2*anchors_per_loc, 1, use_bias=False, name='anchor_logits')
        self.anchor_boxes = tf.keras.layers.Conv2D(
            4*anchors_per_loc, 1, use_bias=False, name='anchor_boxes')

    def call(self, input_data, training=False):
        """
        Parameters
        ----------
        input_data : tensor
            Feature map batch, either in NHWC or NCHW format, depending on how the 
            network was initialized. Should be the ouput of thepenultimate layer 
            in a classifier backbone network. 
        training : bool
            Indicates whether we are in training mode, which affects whether we
            update the mean and variance of the batchnorm layer. Defaults to False.

        Returns
        -------
        list of tensors
            logits, probabilities, and bounding boxes for each anchor, dimensions
            [N, num_anchors, 2], [N, num_anchors, 2], [N, num_anchors, 4],
            respectively.
        """
        num_batch = tf.shape(input_data)[0]
        x = self.shared(input_data)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)

        logits = self.anchor_logits(x)
        boxes = self.anchor_boxes(x)

        logits = tf.reshape(logits, [num_batch, -1, 2])
        probs =  tf.nn.softmax(logits)
        boxes = tf.reshape(boxes, [num_batch, -1, 4])

        return [logits, probs, boxes]

class ProposalLayer(tf.keras.Model):
    """Select a subset of region proposals from the RPN to pass as proposals to
    the second classifier stage.

    Filtering is based on anchor foreground score relative to a threshold, and 
    non-max suppression is used to reduce bounding box overlaps. Bounding box
    refinement is also applied at this stage.

    Parameters
    ----------
    num_proposals : int
        Maximum number of proposals output by the layer. Defaults to 500.
    overlap_thresh : float
        Threshold for deciding if proposals overlap (with respect to the IoU
        metric), during the non-max suppression stage. Defaults to 0.7.
    """
    def __init__(self, num_proposals=500, overlap_thresh=0.7):
        super().__init__()
        self.num_proposals = num_proposals
        self.overlap_thresh = overlap_thresh

    def call(self, input_data):
        """
        Parameters
        ----------
        input_data : list of tensors
            Output of a region proposal network. In order:
            - RPN background/foreground probabilities,
              [N, anchors, (bg prob, fg prob)]
            - RPN bounding box deltas, [N, anchors, (dy, dx, log(dh), log(dw))]
            - Initial bounding box (anchor) estimates, in normalized coordinates,
              [N, anchors, (y1, x1, y2, x2)]
        Returns
        -------
        tensor
            ROI proposals in normalized coordinates [N, roi, (y1, x1, y2, x2)]
        """
        fg_scores = input_data[0][:, :, 1]
        bbox_deltas = input_data[1]
        anchors = input_data[2]
        num_batch = tf.shape(fg_scores)[0]

        updated_boxes = tf.map_fn(lambda x: update_bboxes(x[0], x[1]),
                                  [anchors, bbox_deltas], dtype=tf.float32)
        updated_boxes = tf.clip_by_value(
            updated_boxes, 0, 1, name='updated_bboxes_clipped')

        def nms(x):
            boxes, scores = x[0], x[1]
            nms_indices = tf.image.non_max_suppression(
                boxes, scores, self.num_proposals, 
                iou_threshold=self.overlap_thresh, name='roi_non_max_suppression')
            proposals = tf.gather(boxes, nms_indices)
            padding = tf.maximum(self.num_proposals - tf.shape(proposals)[0], 0)
            proposals = tf.pad(proposals, [(0, padding), (0, 0)])
            return proposals

        final_proposals = tf.map_fn(nms, [updated_boxes, fg_scores], dtype=tf.float32)
        return final_proposals

