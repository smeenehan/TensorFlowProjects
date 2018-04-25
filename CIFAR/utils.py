import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.python.keras.datasets import cifar10

CIFAR10_CLASSES = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', \
                   'horse', 'ship', 'truck']

def device_and_data_format():
  return ('/gpu:0', 'channels_first') if tfe.num_gpus() \
                                      else ('/cpu:0', 'channels_last')

def get_CIFAR10_data(data_format, batch_size=64, shuffle=1000, num_val=1000):
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
    num_val : int
        Number of training images to use as a validation set. Defaults to 1000.

    Returns
    -------
    Dataset
        Training, validation, and test datasets
    Array
        Raw, unprocessed test images as a NumPy array.
    """
    train, val, test, x_test_raw = load_CIFAR10_data(data_format, num_val=num_val)

    # Normalization pre-processing
    mean_image = np.mean(train[0], axis=0)
    def normalize(features, labels):
        return features-mean_image, labels

    train_data = make_generator_dataset(train[0], train[1], batch_size, 
                                        shuffle=shuffle, preprocess_fn=normalize)
    val_data = make_generator_dataset(val[0], val[1], batch_size, 
                                      preprocess_fn=normalize)
    test_data = make_generator_dataset(test[0], test[1], batch_size, 
                                       preprocess_fn=normalize)

    return train_data, val_data, test_data, x_test_raw

def load_CIFAR10_data(data_format, num_val=1000):
    """Load the CIFAR-10 from disk, separating validation and training data

    Parameters
    ----------
    data_format : string
        Whether to Should be 'channels_first' or 'channels_last'
    num_val : int
        Number of training images to use as a validation set. Defaults to 1000.

    Returns
    -------
    Arrays
        (x_train, y_train), (x_val, y_val), (x_test, y_test), x_test_raw
    """
    (x_training, y_training), (x_test, y_test) = cifar10.load_data()

    x_training, x_test = x_training.astype('float32'), x_test.astype('float32')
    y_training = np.squeeze(y_training).astype('int32')
    y_test = np.squeeze(y_test).astype('int32')
    x_test_raw = x_test.copy()

    # Optimize channel ordering for CPU/GPU
    if data_format is 'channels_first':
        x_training = x_training.transpose(0, 3, 1, 2).copy()
        x_test = x_test.transpose(0, 3, 1, 2).copy()

    # Sub-sample training data
    num_train = y_training.shape[0]-num_val
    train_mask = range(num_train)
    val_mask = range(num_train, num_train+num_val)
    x_train, y_train = x_training[train_mask], y_training[train_mask]
    x_val, y_val = x_training[val_mask], y_training[val_mask]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test), x_test_raw

def make_generator_dataset(features, labels, batch_size, shuffle=None, 
                           preprocess_fn=None):
    def gen():
        for image, label in zip(features, labels):
            yield image, label

    data = tf.data.Dataset.from_generator(gen, (tf.float32, tf.int32),
                                          (features[0,:].shape, labels[0].shape))
    data = data.batch(batch_size)
    if shuffle is not None:
       data = data.shuffle(buffer_size=shuffle)
    if preprocess_fn is not None:
       data = data.map(preprocess_fn)
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