import functools
import tensorflow as tf

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
        Number of classes. Defaults to 1001.
    regularizer : function
        Regularizer function applied to all weights in the network. Defaults to
        None.
    include_fc : bool
        If False, the final average pooling and fully connected layers are omitted.
        Use this if we wish to use the raw feature maps at the final stage, e.g.,
        for object detection. Defaults to True.
    """
    def __init__(self, data_format, init_channels=256, num_stages=4, num_blocks=3,
                 classes=1001, regularizer=None, include_fc=True):
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
        if include_fc:
            reduction_indices = [2, 3] if data_format is 'channels_first' else [1, 2]
            reduction_indices = tf.constant(reduction_indices)
            self.global_pool = functools.partial(tf.reduce_mean,
                reduction_indices=reduction_indices, keepdims=False)
            self.fc = tf.keras.layers.Dense(classes, name='fc', 
                                            kernel_regularizer=regularizer)
        else:
            self.global_pool = None
            self.fc = None

    def call(self, input_data, training=False):
        """
        Parameters
        ----------
        input_data : tensor
            Image batch, either in NHWC or NCHW format, depending on how the 
            network was initialized.
        training : bool
            Indicates whether we are in training mode, which affects whether we
            update the mean and variance of the batchnorm layers. Defaults to False.

        Returns
        -------
        tensor
            Class predictions if include_fc was True. Otherwise, the final set of 
            feature maps after batchnorm+ReLU.
        """
        x = self.conv_init(input_data)
        x = self.pool(x)

        for layer in self.layers[2:self.num_res+2]:
            x = layer(x, training=training)

        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        if self.fc is not None:
            x = self.global_pool(x)
            x = self.fc(tf.layers.flatten(x))
        return x

class ResBlock(tf.keras.Model):
    """Implement a residual block.

    Each block has the following form:
        input [NxNxC0] -> (1x1, C1)->(3x3, C1)->(1x1, C2) -> conv+input [MxMxC2]
    with batch-norm and ReLU nonlinearity inserted before each conv operation.

    Note that M != N. If we specify a non-unity stride, M = N/stride, where 
    downsampling via strided convolution occurs on the 3x3 convolution and
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
