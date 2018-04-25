from custom_session_run_hooks import CustomCheckpointSaverHook
import functools
import tensorflow as tf

def compute_accuracy(labels, logits):
    predictions = tf.argmax(logits, axis=1, output_type=tf.int32)
    equality = tf.equal(labels, predictions)
    return tf.reduce_mean(tf.cast(equality, dtype=tf.float32))

def model_fn(features, labels, mode, params, config):
    """Model function for use with a TensorFlow Estimator, implementing a 
    ResNet-style classifier trained with Adam and L2 regularization.

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
    model = ResNet(params['data_format'], 
                   regularizer=tf.keras.regularizers.l2(l=params['reg_scale']))
    image = features
    if isinstance(image, dict):
        image = features['image']

    training = mode is tf.estimator.ModeKeys.TRAIN
    logits = model(image, training=training)
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

    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
        with tf.control_dependencies(extra_update_ops):
            train_op = optimizer.minimize(loss, tf.train.get_or_create_global_step())

    train_spec = tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Use a custom checkpoint saver hook to avoid saving the graph definition 
    # every single checkpoint
    save_hook = CustomCheckpointSaverHook(
        params['model_dir'], save_secs=config.save_checkpoints_secs,
        save_steps=config.save_checkpoints_steps, scaffold=train_spec.scaffold)
    train_spec = train_spec._replace(training_chief_hooks=[save_hook])

    return train_spec


class ResNet(tf.keras.Model):
    """Implement a wide ResNet-style architecture.

    Stacks multiple stages of residual blocks, with a fixed number of residual
    blocks (3) per stage. At each stage, we downsample by a factor of two, using
    strided convolution, and increase the number of channels by two.

    To start we will perform a 3x3 convolution on the input and increase the 
    number of channels to some amount. At the end we'll perform a global average
    pool and a single fully-connected layer.

    Parameters
    ----------
    data_format : string
        'channels_first' or 'channels_last', indicating the ordering of feature
        maps and channels.
    init_channels : int
        Number of channels used in the first convolution. Defaults to 16.
    num_stages : int
        Number of residual stages. Note that we downsample by two after each 
        stage, so this should not be more than log2(M), where M is image dimension.
        Defaults to 3.
    classes : int
        Number of classes. Defaults to 10.
    regularizer : function
        Regularizer function applied to all weights in the network. Defaults to
        None.
    """
    def __init__(self, data_format, init_channels=16, num_stages=3, classes=10,
                 regularizer=None):
        super().__init__()
        self.conv_init = tf.keras.layers.Conv2D(
            init_channels, (3, 3), data_format=data_format, padding='same', 
            name='conv_init', kernel_regularizer=regularizer)

        self.num_stages = num_stages
        output_channels = 2*init_channels
        for idx in range(self.num_stages):
            stage = str(idx)
            self.layers.append(ResBlock(output_channels, data_format, 
                                        stage, 'a', regularizer=regularizer))
            self.layers.append(ResBlock(output_channels, data_format, 
                                        stage, 'b', regularizer=regularizer))
            self.layers.append(ResBlock(output_channels, data_format, 
                                        stage, 'c', regularizer=regularizer))

            output_channels *= 2

        reduction_indices = [2, 3] if data_format is 'channels_first' else [1, 2]
        reduction_indices = tf.constant(reduction_indices)
        self.global_pool = functools.partial(tf.reduce_mean,
            reduction_indices=reduction_indices, keepdims=False)
        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(classes, name='fc', 
                                        kernel_regularizer=regularizer)

    def call(self, input_data, training=False):
        x = self.conv_init(input_data)

        for layer in self.layers[1:3*self.num_stages+1]:
            x = layer(x, training=training)

        x = self.global_pool(x)
        return self.fc(self.flatten(x))

class ResBlock(tf.keras.Model):
    """Implement a residual block.

    Each block has the following form
        input [NxNxC0] -> (1x1, C1)->(3x3, C1)->(1x1, C2) -> conv+input [NxNxC2]
    with batch-norm and ReLU nonlinearity inserted before each conv operation.

    Note that on the first block of each stage, we downsample by two and increase
    output chans. This means that the shortcut path for the first block contains
    a strided convolution and increase in channel number, but there is no batch-norm
    or nonlinearity on this path. The first convolution in the block will also be
    strided in this case.

    Parameters
    ----------
    output_channels : int
        Number of channels in the output.
    data_format : string
        'channels_first' or 'channels_last', indicating the ordering of feature
        maps and channels.
    stage : string
        Name of the residual stage to which this block belongs.
    block : string
        Name of the block within the stage. The first block is expected to be
        named 'a', which will cause us to downsample the input by 2.
    regularizer : function
        Regularizer function applied to all weights in the network. Defaults to
        None.
    """
    def __init__(self, output_channels, data_format, stage, block, regularizer=None):
        super().__init__()
        mid_channels = output_channels//2
        bn_axis = 1 if data_format is 'channels_first' else 3
        bn_name = 'bn'+stage+block+'_'
        conv_name = 'conv'+stage+block+'_'
        # downsample on first block in each layer
        self.first_block = block is 'a'
        strides = (2, 2) if self.first_block else (1, 1)

        self.bn1 = tf.keras.layers.BatchNormalization(axis=bn_axis, name=bn_name+'1')
        self.conv1 = tf.keras.layers.Conv2D(mid_channels, (1, 1), strides=strides, 
                                   name=conv_name+'1', data_format=data_format,
                                   kernel_regularizer=regularizer)

        self.bn2 = tf.keras.layers.BatchNormalization(axis=bn_axis, name=bn_name+'2')
        self.conv2 = tf.keras.layers.Conv2D(mid_channels, (3, 3), name=conv_name+'2', 
                                   padding='same', data_format=data_format,
                                   kernel_regularizer=regularizer)

        self.bn3 = tf.keras.layers.BatchNormalization(axis=bn_axis, name=bn_name+'3')
        self.conv3 = tf.keras.layers.Conv2D(output_channels, (1, 1), name=conv_name+'3', 
                                   data_format=data_format,
                                   kernel_regularizer=regularizer)

        if self.first_block:
            self.conv0 = tf.keras.layers.Conv2D(output_channels, (1, 1),
                strides=strides, name=conv_name+'0', data_format=data_format,
                kernel_regularizer=regularizer)

    def call(self, input_data, training=False):
        x = self.bn1(input_data, training=training)
        x = tf.nn.relu(x)
        x = self.conv1(x)

        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)

        x = self.bn3(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv3(x)

        if self.first_block:
            shortcut = self.conv0(input_data)
        else:
            shortcut = input_data
        return x+shortcut
