from custom_session_run_hooks import CustomCheckpointSaverHook
import functools
from itertools import product
import tensorflow as tf

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
    data_format = params.get('data_format', 'channels_first')
    reg_scale = params.get('reg_scale', 0.0001)
    num_blocks = params.get('num_stages', 3)
    num_blocks = params.get('num_blocks', 3)
    init_channels = params.get('init_channels', 64)
    regularizer = tf.keras.regularizers.l2(l=reg_scale)
    model = ResNet(data_format, init_channels=init_channels, 
                   num_blocks=num_blocks, regularizer=regularizer)
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

    optim_type = params.get('optim_type', 'Adam')
    learning_rate = params.get('learning_rate', 0.01)
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    global_step = tf.train.get_or_create_global_step()
    with tf.name_scope('train'):
        if optim_type is 'Adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        elif optim_type is 'Momentum':
            momentum = params.get('momentum', 0.9)
            if isinstance(learning_rate, list):
                boundaries = learning_rate[0]
                values = learning_rate[1]
                mom_learn_rate = tf.train.piecewise_constant(
                    global_step, boundaries, values)
            else:
                mom_learn_rate = learning_rate
            optimizer = tf.train.MomentumOptimizer(
                learning_rate=mom_learn_rate, momentum=momentum, use_nesterov=True)
        else:
            raise ValueError('Unrecognized optimizer type:', optim_type)

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


class ResNet(tf.keras.Model):
    """Implement a ResNet architecture.

    Stacks multiple stages of residual blocks, with a variable number of residual
    blocks per stage. We always start with a 7x7 strided convolution expanding to
    init_channels/4 channels, followed by a 3x3 max-pooling with stride 2. Then, 
    we pass through a series of residual blocks. The number of output channels 
    is specified for the first stage, and no downsampling is performed. Following 
    this, the output channels are doubled each stage and the feature maps are 
    downsampled by a factor of two.

    Parameters
    ----------
    data_format : string
        'channels_first' or 'channels_last', indicating the ordering of feature
        maps and channels.
    init_channels : int
        Number of channels used in the first residual stage. Defaults to 256.
    num_stages : int
        Number of residual stages. Note that we downsample by two after each 
        stage past the first, so this should not be more than 1+log2(M), where M 
        is image dimension. Defaults to 3.
    num_blocks : int or list of ints
        Number of residual blocks per stage. If this is a list it must be of length
        num_stages. Defaults to 3.
    classes : int
        Number of classes. Defaults to 1000.
    regularizer : function
        Regularizer function applied to all weights in the network. Defaults to
        None.
    """
    def __init__(self, data_format, init_channels=256, num_stages=4, num_blocks=3,
                 classes=1001, regularizer=None):
        super().__init__()
        self.conv_init = tf.keras.layers.Conv2D(
            init_channels//4, 7, strides=2, data_format=data_format, use_bias=False, 
            padding='same', name='conv_init', kernel_regularizer=regularizer)
        self.pool = tf.keras.layers.MaxPool2D(3, strides=2, padding='same', 
                                              data_format=data_format)

        if not isinstance(num_blocks, list):
            num_blocks = [num_blocks]*num_stages
        elif len(num_blocks) is not num_stages:
            raise ValueError('num_blocks must be an integer or list of size = \
                             num_stages')
        self.num_res = 0

        for stage in range(num_stages):
            for block in range(num_blocks[stage]):
                self.num_res += 1
                stride = 2 if (stage>0 and block==0) else 1
                shortcut_activate = True if block==0 else False
                output_channels = init_channels*(2**stage)
                self.layers.append(ResBlock(output_channels, data_format,
                                            shortcut_activate=shortcut_activate,
                                            stride=stride, regularizer=regularizer))


        bn_axis = 1 if data_format is 'channels_first' else 3
        self.bn = tf.layers.BatchNormalization(axis=bn_axis, name='bn_final')
        reduction_indices = [2, 3] if data_format is 'channels_first' else [1, 2]
        reduction_indices = tf.constant(reduction_indices)
        self.global_pool = functools.partial(tf.reduce_mean,
            reduction_indices=reduction_indices, keepdims=False)
        self.fc = tf.keras.layers.Dense(classes, name='fc', 
                                        kernel_regularizer=regularizer)

    def call(self, input_data, training=False):
        x = self.conv_init(input_data)
        x = self.pool(x)

        for layer in self.layers[2:self.num_res+2]:
            x = layer(x, training=training)

        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        x = self.global_pool(x)
        return self.fc(tf.layers.flatten(x))

class ResBlock(tf.keras.Model):
    """Implement a residual block.

    Each block has the following form:
        input [NxNxC0] -> (1x1, C1)->(3x3, C1)->(1x1, C2) -> conv+input [MxMxC2]
    with batch-norm and ReLU nonlinearity inserted before each conv operation.

    Note that M != N. If we specify a non-unity stride, M = N/stride, where 
    downsampling via strided convolution occurs on the first 1x1 convolution and
    the shortcut path. Also note that for now we have a fixed relation C1 = C2/4.
    
    For the 

    Parameters
    ----------
    output_channels : int
        Number of channels in the output (e.g., C2)
    data_format : string
        'channels_first' or 'channels_last', indicating the ordering of feature
        maps and channels.
    stride : int
        Stride used to downsample the input on the first convolution and shortcut
        path. Defaults to 1 (no downsampling).
    shortcut_activate : bool
        Whether or not to make the first activation (batch-norm+ReLU) common to
        the residual and shortcut paths and place a 1x1 (possibly strided) 
        convolution on the shortcut path. Defaults to False (only on residual path).
    regularizer : function
        Regularizer function applied to all weights in the network. Defaults to
        None.
    """
    def __init__(self, output_channels, data_format, stride=1, 
                 shortcut_activate=False, regularizer=None):
        super().__init__()
        mid_channels = output_channels//4
        bn_axis = 1 if data_format is 'channels_first' else 3
        bn_name = 'bn_'
        conv_name = 'conv_'
        self.shortcut_activate = shortcut_activate
        self.stride = stride
        conv_stride = (stride, stride)

        self.bn1 = tf.layers.BatchNormalization(axis=bn_axis, name=bn_name+'1')
        self.conv1 = tf.keras.layers.Conv2D(
            mid_channels, (1, 1), name=conv_name+'1', use_bias=False,
            data_format=data_format, kernel_regularizer=regularizer)

        self.bn2 = tf.layers.BatchNormalization(axis=bn_axis, name=bn_name+'2')
        self.conv2 = tf.keras.layers.Conv2D(
            mid_channels, (3, 3), name=conv_name+'2', padding='same', use_bias=False,
            strides=conv_stride, data_format=data_format, kernel_regularizer=regularizer)

        self.bn3 = tf.layers.BatchNormalization(axis=bn_axis, name=bn_name+'3')
        self.conv3 = tf.keras.layers.Conv2D(
            output_channels, (1, 1), name=conv_name+'3', data_format=data_format,
            use_bias=False, kernel_regularizer=regularizer)

        if self.shortcut_activate:
            self.conv0 = tf.keras.layers.Conv2D(
                output_channels, (1, 1), strides=conv_stride, name=conv_name+'0', 
                use_bias=False, data_format=data_format, kernel_regularizer=regularizer)

    def call(self, input_data, training=False):
        x_in = self.bn1(input_data, training=training)
        x_in = tf.nn.relu(x_in)
        x = self.conv1(x_in)

        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)

        x = self.bn3(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv3(x)

        if self.shortcut_activate:
            shortcut = self.conv0(x_in)
        else:
            shortcut = input_data
        return x+shortcut
