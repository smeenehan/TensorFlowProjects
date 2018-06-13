from custom_session_run_hooks import CustomCheckpointSaverHook
from network.backbone import ResNet
from network.losses import bbox_loss, class_loss
from network.region_proposal import RPN, ProposalLayer
from network.roi_classifier import ROIAlign, ROIHead, DetectionLayer
from network.targets import RPNTargetLayer, DetectionTargetLayer
from network.utils import generate_anchors, update_bboxes_batch
from resnet_var_dict import resnet_var_dict
import tensorflow as tf

def model_fn(features, labels, mode, params, config):
    """
    Parameters
    ----------
    features : tensor 
        Image batch, in NHWC format.
    labels : tensor
        Ground-truth classes and bounding boxes for the image batch.
    mode : string
        ModeKey specifying whether we are in training/evaluation/prediction mode.
    params : dictionary
        Hyperparameters for the model.
    config : config object
        Runtime configuration of the Estimator calling us. Used to set up custom
        saver hooks.

    Returns
    -------
    EstimatorSpec
    """
    images = features
    true_classes = labels['classes']
    true_bboxes = labels['bboxes']
    training = mode is tf.estimator.ModeKeys.TRAIN
    predict = mode is tf.estimator.ModeKeys.PREDICT
    outputs = build_network(images, true_classes, true_bboxes, params, training,
                            predict)
    
    if predict:
        detections = outputs['detect']
        predictions = {'bboxes': detections[:, :, :4],
                       'classes': detections[:, :, 4],
                       'probabilities': detections[:, :, 5]}
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = setup_loss(outputs)

    if mode is tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss)

    init_backbone(params)

    learning_rate = params.get('learning_rate', 0.01)
    momentum = params.get('momentum', 0.9)
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    global_step = tf.train.get_or_create_global_step()
    with tf.name_scope('train'):
        if isinstance(learning_rate, list):
            boundaries = learning_rate[0]
            values = learning_rate[1]
            learn_rate = tf.train.piecewise_constant(
                global_step, boundaries, values)
        else:
            learn_rate = learning_rate
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learn_rate, momentum=momentum, use_nesterov=True)

        with tf.control_dependencies(extra_update_ops):
            train_op = optimizer.minimize(loss, global_step)

    train_spec = tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Use a custom checkpoint saver hook to avoid saving the graph definition 
    # every single checkpoint
    save_hook = CustomCheckpointSaverHook(
        config.model_dir, save_secs=config.save_checkpoints_secs,
        save_steps=config.save_checkpoints_steps, scaffold=train_spec.scaffold)
    train_spec = train_spec._replace(training_chief_hooks=[save_hook])

    return train_spec

def build_network(images, true_classes, true_bboxes, params, training, predict):
    """
    Parameters
    ----------
    images : tensor 
        Image batch, in NHWC format.
    true_classes : tensor
        Ground-truth classes and bounding boxes for the image batches
    true_bboxes : string
        ModeKey specifying whether we are in training/evaluation/prediction mode.
    params : dictionary
        Hyperparameters for the model.
    training : bool
        True if we are in training mode.
    predict : bool
        True if we are in prediction mode.

    Returns
    -------
    dictionary of tensors
        Always contains the keys 'rpn' = [logits, deltas] from the region 
        proposal stage, and 'roi' = [target_boxes, logits, deltas] from the ROI 
        classier stage. In training mode, also includes 'rpn_targets' and 
        'roi_targets', each of which is a list of target classes and box deltas 
        for the two stages. In prediction mode, contains the key 'detect', which
        returns the filtered detections from the ROI classifier stage.
    """
    backbone = ResNet('channels_last', num_blocks=[3, 4, 6, 3], include_fc=False)
    feature_maps = backbone(images, training=training)

    anchor_stride = params.get('anchor_stride', 1)
    anchor_scales = params.get('anchor_scales', [32, 64, 128])
    anchor_ratios = params.get('anchor_ratios', [0.5, 1.0, 1.5])
    num_anchors = len(anchor_scales)*len(anchor_ratios)
    rpn_graph = RPN('channels_last', anchors_per_loc=num_anchors, 
                    anchor_stride=anchor_stride)
    rpn_logits, rpn_probs, rpn_deltas = rpn_graph(feature_maps, training=training)
    return_dict = {'rpn': [rpn_logits, rpn_deltas]}

    with tf.name_scope('anchor_gen'):
        anchors = generate_anchors(
            anchor_scales, anchor_ratios, tf.shape(images)[1:3], 
            tf.shape(feature_maps)[1:3], anchor_stride)
        anchors = tf.tile(tf.expand_dims(anchors, 0), [tf.shape(images)[0], 1, 1])

    proposer = ProposalLayer()
    roi_proposals = proposer([rpn_probs, rpn_deltas, anchors])

    if training:
        rpn_target = RPNTargetLayer()
        rpn_target_classes, rpn_target_deltas = rpn_target([anchors, true_bboxes])

        det_target = DetectionTargetLayer()
        roi_targets, roi_target_classes, roi_target_deltas = det_target(
            [roi_proposals, true_classes, true_bboxes])

        return_dict['rpn_targets'] = [rpn_target_classes, rpn_target_deltas]
        return_dict['roi_targets'] = [roi_target_classes, roi_target_deltas]
    else:
        roi_targets = roi_proposals

    roi_align = ROIAlign()
    roi_features = roi_align([feature_maps, roi_targets])

    roi_head = ROIHead('channels_last')
    roi_logits, roi_probs, roi_deltas = roi_head(roi_features)
    return_dict['roi'] = [roi_targets, roi_logits, roi_deltas]

    if predict:
        detect = DetectionLayer()
        detected_objects = detect([roi_targets, roi_probs, roi_deltas])
        return_dict['detect'] = detected_objects

    return return_dict

def setup_loss(outputs):
    rpn_logits, rpn_deltas = outputs['rpn']
    rpn_target_classes, rpn_target_deltas = outputs['rpn_targets']
    with tf.name_scope('rpn_loss'):
        rpn_class_loss = class_loss(rpn_logits, rpn_target_classes)
        rpn_bbox_loss = bbox_loss(rpn_deltas, rpn_target_classes, rpn_target_deltas)
        rpn_loss = rpn_class_loss+10*rpn_bbox_loss
    tf.summary.scalar('rpn_loss', rpn_loss)

    roi_targets, roi_logits, roi_deltas = outputs['roi']
    roi_target_classes, roi_target_deltas = outputs['roi_targets']
    with tf.name_scope('reduce_roi_deltas'):
        shape = tf.shape(roi_deltas)
        roi_ids = tf.argmax(roi_logits, axis=2, output_type=tf.int32)
        batch_inds = tf.tile(tf.range(shape[0])[:, None, None], [1, shape[1], shape[3]])
        roi_inds = tf.tile(tf.range(shape[1])[None, :, None], [shape[0], 1, shape[3]])
        class_inds = tf.tile(roi_ids[:, :, None], [1, 1, shape[3]])
        delta_inds = tf.tile(tf.range(shape[3])[None, None, :], [shape[0], shape[1], 1])
        indices = tf.stack([batch_inds, roi_inds, class_inds, delta_inds], axis=3)
        reduced_roi_deltas = tf.gather_nd(roi_deltas, indices)

    with tf.name_scope('roi_loss'):
        roi_class_loss = class_loss(roi_logits, roi_target_classes)
        roi_bbox_loss = bbox_loss(reduced_roi_deltas, roi_target_classes, 
                                  roi_target_deltas)
        roi_loss = roi_class_loss+10*roi_bbox_loss
    tf.summary.scalar('roi_loss', roi_loss)

    with tf.name_scope('total_loss'):
        total_loss = rpn_loss+roi_loss+tf.losses.get_regularization_loss()
    tf.summary.scalar('total_loss', total_loss)
    return total_loss

def init_backbone(params, backbone_name='res_net'):
    backbone_ckpt = params.get('backbone_ckpt', None)
    if backbone_ckpt is None:
        return
    backbone_var_dict = {}
    for key, item in resnet_var_dict.items():
        if 'dense' in key:
            continue
        new_item = item.replace('res_net', backbone_name)
        backbone_var_dict[key] = new_item
    tf.train.init_from_checkpoint(backbone_ckpt, backbone_var_dict)