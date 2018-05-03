import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.python.keras.datasets import cifar10

CIFAR10_CLASSES = ['plane', 'car', 'bird', 'cat', 'deer', \
                   'dog', 'frog', 'horse', 'ship', 'truck']

def device_and_data_format():
  return ('/gpu:0', 'channels_first') if tfe.num_gpus() \
                                      else ('/cpu:0', 'channels_last')

def get_CIFAR10_data(data_format, batch_size=64, shuffle=1000, augment=False,
                     num_val=0):
    """Load the CIFAR-10 dataset, separate training set into training/validation,
    and pre-process the data while keeping the raw test-data for display. 

    Parameters
    ----------
    data_format : string
        Whether to Should be 'channels_first' or 'channels_last'
    batch_size : int
        Minibatch size for SGD. Defaults to 64.
    shuffle : int
        Size of the shuffle buffer when sampling mini-batches. Defaults to 1000.
    augment : bool
        Whether or not to augment the training set with random crops and flips.
        Defaults to False.
    num_val : int
        Number of training images to use as a validation set. Defaults to 0.

    Returns
    -------
    Dataset
        Training, test, and validation datasets. Validation dataset can be None
    """
    train, test, val, _ = load_CIFAR10_data(data_format, num_val=num_val)

    # Normalization pre-processing
    mean_image = np.mean(train[0], axis=0)
    def normalize(image, label):
        image = image-mean_image
        if data_format is 'channels_first':
            image = tf.transpose(image, perm=[2, 0, 1])
        return image, label#-mean_image, label

    def augment_data(image, label):
        image = image-mean_image
        image = tf.image.resize_image_with_crop_or_pad(image, 40, 40)
        image = tf.random_crop(image, [32, 32, 3])
        image = tf.image.random_flip_left_right(image)
        if data_format is 'channels_first':
            image = tf.transpose(image, perm=[2, 0, 1])
        return image, label

    train_fn = augment_data if augment else normalize

    train_data = make_generator_dataset(train[0], train[1], batch_size, 
                                        shuffle=shuffle, preprocess_fn=train_fn)
    test_data = make_generator_dataset(test[0], test[1], batch_size, 
                                       preprocess_fn=normalize)
    if val is not None:
        val_data = make_generator_dataset(val[0], val[1], batch_size, 
                                          preprocess_fn=normalize)
    else:
        val_data = None

    return train_data, test_data, val_data

def load_CIFAR10_data(data_format, num_val=0):
    """Load the CIFAR-10 from disk, separating validation and training data

    Parameters
    ----------
    data_format : string
        Whether to Should be 'channels_first' or 'channels_last'
    num_val : int
        Number of training images to use as a validation set. Defaults to 0.

    Returns
    -------
    Arrays
        (x_train, y_train), (x_test, y_test), (x_val, y_val), x_test_raw
    """
    (x_training, y_training), (x_test, y_test) = cifar10.load_data()

    x_training, x_test = x_training.astype('float32'), x_test.astype('float32')
    y_training = np.squeeze(y_training).astype('int32')
    y_test = np.squeeze(y_test).astype('int32')
    x_test_raw = x_test.copy()

    # Sub-sample training data if we want a validation split
    if num_val > 0:
        num_train = y_training.shape[0]-num_val
        train_mask = range(num_train)
        val_mask = range(num_train, num_train+num_val)
        x_train, y_train = x_training[train_mask], y_training[train_mask]
        val = (x_training[val_mask], y_training[val_mask])
    else:
        x_train, y_train, val = x_training, y_training, None

    return (x_train, y_train), (x_test, y_test), val, x_test_raw

def make_generator_dataset(features, labels, batch_size, shuffle=None, 
                           preprocess_fn=None):
    def gen():
        for image, label in zip(features, labels):
            yield image, label

    data = tf.data.Dataset.from_generator(gen, (tf.float32, tf.int32),
                                          (features[0,:].shape, labels[0].shape))
    if preprocess_fn is not None:
       data = data.map(preprocess_fn)
    data = data.batch(batch_size)
    if shuffle is not None:
       data = data.shuffle(buffer_size=shuffle)
    return data

def show_random_image(images, labels, preds=None):
    rand_idx = np.random.choice(labels.shape[0])
    show_image_with_label(images[rand_idx], labels[rand_idx])
    if preds is not None:
        class_pred = CIFAR10_CLASSES[preds[rand_idx][0]]
        print('Prediction:', class_pred)


def show_image_with_label(image, label):
    plt.imshow(image.astype('uint8'), interpolation='nearest')
    plt.show()
    class_pred = CIFAR10_CLASSES[label[0]]
    print('Truth:', class_pred)