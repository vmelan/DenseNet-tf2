import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers

import numpy as np
import pdb

class BottleneckLayer(layers.Layer):
    def __init__(self, growthRate):
        super().__init__()
        self.conv1 = layers.Conv2D(4 * growthRate, kernel_size=1, strides=1, padding="same")
        self.conv2 = layers.Conv2D(growthRate, kernel_size=3, strides=1, padding="same")
        self.batchNorm = layers.BatchNormalization(momentum=0.99, epsilon=0.001)
        self.relu = layers.Activation("relu")

    def call(self, x):
        y = self.batchNorm(self.relu(self.conv1(x)))
        y = self.batchNorm(self.relu(self.conv2(y)))
        y = layers.concatenate([x, y])
        return y

class DenseBlock(layers.Layer):
    def __init__(self, growthRate, numDenseBlocks):
        super().__init__()
        self.growthRate = growthRate
        self.numDenseBlocks = numDenseBlocks

    def call(self, x):
        for _ in range(self.numDenseBlocks):
            x = BottleneckLayer(self.growthRate)(x)
            print(x.shape)
        return x

# class TransitionLayer(layers.Layer):
#     def __init__(self, numChannels, compressionFactor):
#         pass
#
#     def call(self, x):
#         pass



class DenseNet(Model):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.relu = layers.Activation("relu")
        growthRate = 12
        self.conv1 = layers.Conv2D(2 * growthRate, kernel_size=7, strides=2, padding="same")
        self.maxpool = layers.MaxPooling2D((2, 2), strides=2) # paper uses (3,3) kernel but not consistent with output size

        # self.bottleneck = BottleneckLayer(growthRate)
        # self.bottleneck2 = BottleneckLayer(growthRate)

        ## 1st denseblock
        self.denseblock1 = DenseBlock(growthRate, 6)

    def call(self, x):
        y = self.maxpool(self.relu(self.conv1(x)))
        print(y.shape)
        ## putting BottleneckLayer directly in call works
        # for _ in range(6):
        #     y = BottleneckLayer(12)(y)
        #     print(y.shape)

        ## this approach does not work
        # y = self.bottleneck(y)
        # print(y.shape)
        # y = self.bottleneck2(y)
        # print(y.shape)

        # for _ in range(6):
        #     y = self.bottleneck(y)
        #     print(y.shape)

        y = self.denseblock1(y)

        pdb.set_trace()

        return y


## class for debug help
# class DenseNet(Model):
#     def __init__(self, config):
#         super().__init__()
#         self.growthRate = 12
#
#     def call(self, x):
#         y = layers.Conv2D(2 * self.growthRate, kernel_size=7, strides=2, padding="same")(x)
#         y = layers.Activation("relu")(y)
#         x = layers.MaxPooling2D((2, 2), strides=2)(y) # paper uses (3,3) kernel but not consistent with output size
#
#         print(x.shape)
#
#         ## denseblock
#         y = layers.Conv2D(4 * self.growthRate, kernel_size=1, strides=1, padding="same")(x)
#         y = layers.Activation("relu")(y)
#         y = layers.BatchNormalization(momentum=0.99, epsilon=0.001)(y)
#         y = layers.Conv2D(self.growthRate, kernel_size=1, strides=1, padding="same")(y)
#         y = layers.Activation("relu")(y)
#         y = layers.BatchNormalization(momentum=0.99, epsilon=0.001)(y)
#         x = layers.concatenate([x, y])
#
#         print(x.shape)
#
#         y = layers.Conv2D(4 * self.growthRate, kernel_size=1, strides=1, padding="same")(x)
#         y = layers.Activation("relu")(y)
#         y = layers.BatchNormalization(momentum=0.99, epsilon=0.001)(y)
#         y = layers.Conv2D(self.growthRate, kernel_size=1, strides=1, padding="same")(y)
#         y = layers.Activation("relu")(y)
#         y = layers.BatchNormalization(momentum=0.99, epsilon=0.001)(y)
#         x = layers.concatenate([x, y])
#
#         print(x.shape)

if __name__ == "__main__":
    layer = BottleneckLayer(3, 32)
    x = tf.random.uniform((16, 224,224, 3))
    x = layer(3)
    pdb.set_trace()
