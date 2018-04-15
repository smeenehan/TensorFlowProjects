import matplotlib.pyplot as plt
import numpy as np
import tensorflow.contrib.eager as tfe
from tensorflow.python.keras.datasets import cifar10

CIFAR10_CLASSES = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', \
                   'horse', 'ship', 'truck']

def device_and_data_format():
  return ('/gpu:0', 'channels_first') if tfe.num_gpus() \
                                      else ('/cpu:0', 'channels_last')

def get_CIFAR10_data(data_format, num_val=1000):
    """
    Load the CIFAR-10 dataset, separate training set into training/validation,
    and pre-process the data while keeping the raw test-data for display. 
    """
    (x_training, y_training), (x_test, y_test) = cifar10.load_data()

    x_test = x_test.astype('float32')
    x_test_raw = x_test.copy()

    # Optimize channel ordering for CPU/GPU
    if data_format is 'channels_first':
        x_training = x_training.transpose(0, 3, 1, 2).copy()
        x_test = x_test.transpose(0, 3, 1, 2).copy()

    # Sub-sample training data
    num_train = y_training.shape[0]-num_val
    train_mask = range(num_train)
    val_mask = range(num_train, num_train+num_val)
    x_train, y_train = x_training[train_mask].astype('float32'), y_training[train_mask]
    x_val, y_val = x_training[val_mask].astype('float32'), y_training[val_mask]

    # Normalize data
    mean_image = np.mean(x_train, axis=0)
    x_train -= mean_image
    x_val -= mean_image
    x_test -= mean_image

    return x_train, y_train, x_val, y_val, x_test, y_test, x_test_raw

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