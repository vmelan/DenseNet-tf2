import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers

import numpy as np
import pdb

# class BottleneckLayer(layers.Layer):
#     def __init__(self, growthRate):
#         super().__init__()
#         self.conv1 = layers.Conv2D(4 * growthRate, kernel_size=1, strides=1, padding="same")
#         self.conv2 = layers.Conv2D(growthRate, kernel_size=3, strides=1, padding="same")
#         self.batchNorm = layers.BatchNormalization(momentum=0.99, epsilon=0.001)
#         self.relu = layers.Activation("relu")
#
#     def call(self, x):
#         print(x.shape)
#         y = self.batchNorm(self.relu(self.conv1(x)))
#         y = self.batchNorm(self.relu(self.conv2(y)))
#         y = layers.concatenate([x, y])
#         print(y.shape)
#         return y
#
#
# class DenseBlock(layers.Layer):
#     def __init__(self, growthRate, numDenseBlocks):
#         super().__init__()
#         self.growthRate = growthRate
#         self.numDenseBlocks = numDenseBlocks
#
#     def call(self, x):
#         pdb.set_trace()
#         for i in range(self.numDenseBlocks):
#             x = BottleneckLayer(self.growthRate)(x)
#             print(x.shape)
#
#         numChannels = x.shape[-1]
#         return x, numChannels


## Denseblock class without using bottleneck layer class
class DenseBlock(layers.Layer):
    def __init__(self, growthRate, numDenseBlocks):
        super().__init__()
        self.growthRate = growthRate
        self.numDenseBlocks = numDenseBlocks

    def call(self, x):
        # pdb.set_trace()
        for _ in range(self.numDenseBlocks):
            x = BottleneckLayer(self.growthRate)(x)

            print(x.shape)

        numChannels = x.shape[-1]
        return x, numChannels

class BottleneckLayer(layers.Layer):
    def __init__(self, growthRate):
        super().__init__()
        self.growthRate = growthRate

    def call(self, x):
        y = layers.Conv2D(4 * self.growthRate, kernel_size=1, strides=1, padding="same")(x)
        y = layers.Activation("relu")(y)
        y = layers.BatchNormalization(momentum=0.99, epsilon=0.001)(y)
        y = layers.Conv2D(self.growthRate, kernel_size=1, strides=1, padding="same")(y)
        y = layers.Activation("relu")(y)
        y = layers.BatchNormalization(momentum=0.99, epsilon=0.001)(y)
        x = layers.concatenate([x, y])
        return x


class TransitionLayer(layers.Layer):
    def __init__(self, numChannels, compressionFactor):
        super().__init__()
        self.batchNorm = layers.BatchNormalization(momentum=0.99, epsilon=0.001)
        self.conv = layers.Conv2D(int(numChannels * compressionFactor),
                                  kernel_size=1, strides=1, padding="same")
        self.relu = layers.Activation("relu")
        self.avgpool = layers.AveragePooling2D((2, 2), strides=2)

    def call(self, x):
        y = self.batchNorm(self.relu(self.conv(x)))
        y = self.avgpool(y)
        return y


class DenseNet(Model):
    def __init__(self, config):
        super().__init__()
        self.growthRate = config["model"]["growth_rate"]
        self.compressionFactor = config["model"]["compression_factor"]

        if config["model"]["name"] == "DenseNet121":
            numDenseBlocks = [6, 12, 24, 16]
        elif config["model"]["name"] == "DenseNet169":
            numDenseBlocks = [6, 12, 32, 32]
        elif config["model"]["name"] == "DenseNet201":
            numDenseBlocks = [6, 12, 48, 32]
        elif config["model"]["name"] == "DenseNet201":
            numDenseBlocks = [6, 12, 64, 48]

        ## denseblocks
        self.denseblock1 = DenseBlock(self.growthRate, numDenseBlocks[0])
        self.denseblock2 = DenseBlock(self.growthRate, numDenseBlocks[1])
        self.denseblock3 = DenseBlock(self.growthRate, numDenseBlocks[2])
        self.denseblock4 = DenseBlock(self.growthRate, numDenseBlocks[3])


    def call(self, x):

        # initial convolution and pooling
        y = layers.Conv2D(2 * self.growthRate, kernel_size=7, strides=2, padding="same")(x)
        y = layers.Activation("relu")(y)
        y = layers.MaxPooling2D((2, 2), strides=2)(y) # paper uses (3,3) kernel but not consistent with output size


        print(y.shape)
        y, numChannels = self.denseblock1(y)
        print(y.shape)

        y = TransitionLayer(numChannels, self.compressionFactor)(y)
        print("Transition block", y.shape)

        y, numChannels = self.denseblock2(y)
        y = TransitionLayer(numChannels, self.compressionFactor)(y)

        print("Transition block", y.shape)

        y, numChannels = self.denseblock3(y)
        y = TransitionLayer(numChannels, self.compressionFactor)(y)

        print("Transition block", y.shape)

        y, numChannels = self.denseblock4(y)

        print(y.shape)


        pdb.set_trace()

        return y


## class for debug help
# class DenseNet(Model):
#     def __init__(self, config):
#         super().__init__()
#         self.growthRate = 12
#         self.compressionFactor = 0.5
#
#     def call(self, x):
#
#
#         y = layers.Conv2D(2 * self.growthRate, kernel_size=7, strides=2, padding="same")(x)
#         y = layers.Activation("relu")(y)
#         x = layers.MaxPooling2D((2, 2), strides=2)(y) # paper uses (3,3) kernel but not consistent with output size
#
#         print(x.shape)
#
#         ## Denseblock with loop
#         for _ in range(6):
#             y = layers.Conv2D(4 * self.growthRate, kernel_size=1, strides=1, padding="same")(x)
#             y = layers.Activation("relu")(y)
#             y = layers.BatchNormalization(momentum=0.99, epsilon=0.001)(y)
#             y = layers.Conv2D(self.growthRate, kernel_size=1, strides=1, padding="same")(y)
#             y = layers.Activation("relu")(y)
#             y = layers.BatchNormalization(momentum=0.99, epsilon=0.001)(y)
#             x = layers.concatenate([x, y])
#             print(x.shape)
#
#
#         ## transition block
#         numChannels = x.shape[-1]
#         y = layers.BatchNormalization(momentum=0.99, epsilon=0.001)(x)
#         y = layers.Conv2D(int(numChannels * self.compressionFactor),
#                                   kernel_size=1, strides=1, padding="same")(y)
#         y = layers.Activation("relu")(y)
#         x = layers.AveragePooling2D((2, 2), strides=2)(y)
#
#         print("Transition block", x.shape)
#
#
#         ## second denseblock
#         for _ in range(12):
#             y = layers.Conv2D(4 * self.growthRate, kernel_size=1, strides=1, padding="same")(x)
#             y = layers.Activation("relu")(y)
#             y = layers.BatchNormalization(momentum=0.99, epsilon=0.001)(y)
#             y = layers.Conv2D(self.growthRate, kernel_size=1, strides=1, padding="same")(y)
#             y = layers.Activation("relu")(y)
#             y = layers.BatchNormalization(momentum=0.99, epsilon=0.001)(y)
#             x = layers.concatenate([x, y])
#             print(x.shape)
#
#         ## transition block
#         numChannels = x.shape[-1]
#         y = layers.BatchNormalization(momentum=0.99, epsilon=0.001)(x)
#         y = layers.Conv2D(int(numChannels * self.compressionFactor),
#                                   kernel_size=1, strides=1, padding="same")(y)
#         y = layers.Activation("relu")(y)
#         x = layers.AveragePooling2D((2, 2), strides=2)(y)
#
#         print("Transition block", x.shape)
#
#
#         ## Third denseblock
#         for _ in range(24):
#             y = layers.Conv2D(4 * self.growthRate, kernel_size=1, strides=1, padding="same")(x)
#             y = layers.Activation("relu")(y)
#             y = layers.BatchNormalization(momentum=0.99, epsilon=0.001)(y)
#             y = layers.Conv2D(self.growthRate, kernel_size=1, strides=1, padding="same")(y)
#             y = layers.Activation("relu")(y)
#             y = layers.BatchNormalization(momentum=0.99, epsilon=0.001)(y)
#             x = layers.concatenate([x, y])
#             print(x.shape)
#
#         ## transition block
#         numChannels = x.shape[-1]
#         y = layers.BatchNormalization(momentum=0.99, epsilon=0.001)(x)
#         y = layers.Conv2D(int(numChannels * self.compressionFactor),
#                                   kernel_size=1, strides=1, padding="same")(y)
#         y = layers.Activation("relu")(y)
#         x = layers.AveragePooling2D((2, 2), strides=2)(y)
#
#         print("Transition block", x.shape)
#
#
#         ## fourth and last denseblock
#         for _ in range(16):
#             y = layers.Conv2D(4 * self.growthRate, kernel_size=1, strides=1, padding="same")(x)
#             y = layers.Activation("relu")(y)
#             y = layers.BatchNormalization(momentum=0.99, epsilon=0.001)(y)
#             y = layers.Conv2D(self.growthRate, kernel_size=1, strides=1, padding="same")(y)
#             y = layers.Activation("relu")(y)
#             y = layers.BatchNormalization(momentum=0.99, epsilon=0.001)(y)
#             x = layers.concatenate([x, y])
#             print(x.shape)

        ## no transition block after last denseblock


import json
if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)
    model = DenseNet(config)
    x = tf.random.uniform((2, 224,224, 3))
    y = model(x)
    print(y.shape)
    pdb.set_trace()
