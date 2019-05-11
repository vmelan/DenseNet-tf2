import pickle
import numpy as np
import tensorflow as tf

import pdb

class DataLoader():
    """ Load CIFAR dataset """
    def __init__(self, config):
    	# Load config file
        self.config = config

    def unpickle(self, file):
    	with open(file, 'rb') as fo:
    		dict = pickle.load(fo, encoding='bytes')
    	return dict

    def load_cifar10(self, data_path):
        # from https://luckydanny.blogspot.com/2016/07/load-cifar-10-dataset-in-python3.html
        train_data = None
        train_labels = []
        test_data = None
        test_labels = None

        for i in range(1, 6):
        	data_dict = self.unpickle(data_path + "data_batch_" + str(i))
        	if (i == 1):
        		train_data = data_dict[b'data']
        	else:
        		train_data = np.vstack((train_data, data_dict[b'data']))
        	train_labels += data_dict[b'labels']

        test_data_dict = self.unpickle(data_path + "test_batch")
        test_data = test_data_dict[b'data']
        test_labels = test_data_dict[b'labels']

        train_data = train_data.reshape((50000, 3, 32, 32))
        train_data = np.rollaxis(train_data, 1, 4)
        train_labels = np.array(train_labels)

        test_data = test_data.reshape((10000, 3, 32, 32))
        test_data = np.rollaxis(test_data, 1, 4)
        test_labels = np.array(test_labels)

        return train_data, train_labels, test_data, test_labels

    def prepare_data(self):
        """ Prepare data by using TensorFlow data API"""
        X_train, y_train, X_test, y_test = self.load_cifar10(self.config["input"]["data_path"])

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

        train_dataset = train_dataset.map(self.preprocess)
        test_dataset = test_dataset.map(self.preprocess)

        batch_size = self.config["trainer"]["batch_size"]
        buffer_size = self.config["trainer"]["buffer_size"]
        train_loader = train_dataset.shuffle(buffer_size).batch(batch_size)
        test_loader = test_dataset.shuffle(buffer_size).batch(batch_size)

        return train_loader, test_loader


    def preprocess(self, image, label):
        # cast image to float32 type
        image = tf.cast(image, tf.float32)
        # normalize according to training stats
        image = (image - self.config["input"]["mean"]) / self.config["input"]["std"]
        # convert label to one hot
        label = tf.one_hot(label, 10)

        return image, label
