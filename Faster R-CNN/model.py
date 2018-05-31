from backbone import ResNet
from region_proposal import RPN, ProposalLayer
import tensorflow as tf
from utils import generate_anchors

def model_fn(features, labels, mode, params, config):
    """
    Parameters
    ----------
    features : Tensor or dictionary
        Feature or dictionary mapping names to feature Tensors.
    labels : Tensor or dictionary
        Label for the feature, or dictionary mapping names to label Tensors.
    mode : string
        ModeKey specifying whether we are in training/evaluation/prediction mode.
    params : dictionary
        Hyperparameters for the model.
    config : config object
        Runtime configuration of the Estimator calling us. Used to set up custom
        saver hooks

    Returns
    -------
    EstimatorSpec
    """
    data_format = params.get('data_format', 'channels_first')
    anchor_stride = params.get('anchor_stride', 1)
    anchor_scales = params.get('anchor_scales', [32, 64, 128])
    anchor_ratios = params.get('anchor_ratios', [0.5, 1.0, 1.5])
    num_anchors = len(anchor_stride)*len(anchor_ratios)
    backbone = ResNet(data_format, num_blocks=[3, 4, 6, 3], include_fc=False)
    rpn_graph = RPN(data_format, anchors_per_loc=num_anchors, 
                    anchor_stride=anchor_stride)
    proposer = ProposalLayer()

    image = features
    if isinstance(image, dict):
        image = features['image']
    training = mode is tf.estimator.ModeKeys.TRAIN

    feature_maps = backbone(image, training=training)
    rpn_logits, rpn_probs, rpn_boxes = rpn_graph(feature_maps, training=training)
    with tf.name_scope('anchor_gen'):
        anchors = generate_anchors(
            anchor_scales, anchor_ratios, tf.shape(image)[1:3], 
            tf.shape(feature_maps)[1:3], anchor_stride)
        anchors = tf.tile(tf.expand_dims(anchors, 0), [tf.shape(images)[0], 1, 1])
    roi_proposals = proposer([rpn_probs, rpn_boxes, anchors])
    
    classes = tf.argmax(logits, axis=1, output_type=tf.int32, name='predictions')

    if mode is tf.estimator.ModeKeys.PREDICT:
        predictions = {'classes': classes, 'probabilities': tf.nn.softmax(logits)}
        return tf.estimator.EstimatorSpec(
            mode=mode, predictions=predictions)

    with tf.name_scope('loss_op'):
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits) \
              +tf.losses.get_regularization_loss()
    tf.summary.scalar('loss', loss)

    accuracy = tf.metrics.accuracy(labels=labels, 
                                   predictions=classes, name='accuracy_op')
    tf.summary.scalar('accuracy', accuracy[1])

    if mode is tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, 
                                          eval_metric_ops={'accuracy': accuracy})
    
    init_from_checkpoints(params, backbone.name)

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

def init_from_checkpoints(params, backbone_name):
    backbone_ckpt = params.get('backbone_ckpt', None)
    if backbone_ckpt is not None:
        backbone_var_dict = {}
        for key, item in resnet_var_dict.items():
            if 'dense' in key:
                continue
            new_item = item.replace('res_net', backbone_name)
            backbone_var_dict[key] = new_item
        tf.train.init_from_checkpoint(backbone_ckpt, backbone_var_dict)