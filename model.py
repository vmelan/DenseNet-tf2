import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers

import numpy as np
import pdb

class BottleneckLayer(layers.Layer):
    def __init__(self, numChannels, growthRate):
        super().__init__()
        numChannels = 4 * growthRate
        self.conv1 = layers.Conv2D(numChannels, kernel_size=1, strides=1, padding="valid")
        self.conv2 = layers.Conv2D(growthRate, kernel_size=3, strides=1, padding="same")
        self.batchNorm = layers.BatchNormalization(momentum=0.99, epsilon=0.001, name="bn1")
        self.relu = layers.Activation("relu")

    def call(self, x):
        y = self.batchNorm(self.relu(self.conv1(x)))
        y = self.batchNorm(self.relu(self.conv2(y)))
        y = layers.concatenate([x, y])

        return y

# class ConvBNRelu(layers.Layer):
#     def __init__(self):
#         self.conv = layers.Conv2D(num_filters, kernel_size=3,
#                                   strides=1, padding="same")
#
#
#     def call(self, inputs):



class DenseNet(Model):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.relu = layers.Activation("relu")
        growthRate = 12
        numChannels = 2 * growthRate
        self.conv1 = layers.Conv2D(numChannels, kernel_size=7, strides=2, padding="same")
        self.maxpool = layers.MaxPooling2D((2, 2), strides=2)


    def call(self, x):
        y = self.maxpool(self.relu(self.conv1(x)))

        return y


if __name__ == "__main__":
    layer = BottleneckLayer(3, 32)
    x = tf.random.uniform((16, 224,224, 3))
    x = layer(3)
    pdb.set_trace()
