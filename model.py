import tensorflow as tf
# from tensorflow import keras
from tensorflow.keras import Model, layers

import numpy as np
import pdb


## Denseblock class without using bottleneck layer class
# class DenseBlock(Model):
class DenseBlock(layers.Layer):
    def __init__(self, growthRate, numDenseBlocks):
        super().__init__()
        self.growthRate = growthRate
        self.numDenseBlocks = numDenseBlocks

        self.listLayers = []
        for _ in range(numDenseBlocks):
            self.listLayers.append(BottleneckLayer(self.growthRate))

    def call(self, x):
        for layer in self.listLayers.layers:
            x = layer(x)
        numChannels = x.shape[-1]
        return x, numChannels

# class BottleneckLayer(Model):
class BottleneckLayer(layers.Layer):
    def __init__(self, growthRate):
        super().__init__()
        self.growthRate = growthRate

        self.conv1 = layers.Conv2D(4 * self.growthRate, kernel_size=1, strides=1, padding="same")
        self.conv2 = layers.Conv2D(self.growthRate, kernel_size=3, strides=1, padding="same")
        self.bn1 = layers.BatchNormalization(momentum=0.99, epsilon=0.001)
        self.bn2 = layers.BatchNormalization(momentum=0.99, epsilon=0.001)

        self.listLayers = [self.conv1,
                           layers.Activation("relu"),
                           self.bn1,
                           self.conv2,
                           layers.Activation("relu"),
                           self.bn2]

    def call(self, x, dropoutRate=0.2):
        # y = self.conv1(x)
        #
        # # y = layers.Dropout(dropoutRate)(y)
        #
        # y = layers.Activation("relu")(y)
        # y = self.bn1(y)
        # y = self.conv2(y)
        #
        # # y = layers.Dropout(dropoutRate)(y)
        #
        # y = layers.Activation("relu")(y)
        # y = self.bn2(y)
        y = x
        for layer in self.listLayers.layers:
            y = layer(y)

        y = layers.concatenate([x, y])
        return y


# class TransitionLayer(Model):
class TransitionLayer(layers.Layer):
    def __init__(self, numChannels, compressionFactor, dropoutRate=0.2):
        super().__init__()
        self.batchNorm = layers.BatchNormalization(momentum=0.99, epsilon=0.001)
        self.conv = layers.Conv2D(int(numChannels * compressionFactor),
                                  kernel_size=1, strides=1, padding="same")

        self.dropout = layers.Dropout(dropoutRate)

        self.relu = layers.Activation("relu")
        self.avgpool = layers.AveragePooling2D((2, 2), strides=2)

    def call(self, x):
        # y = self.batchNorm(self.relu(self.conv(x)))
        y = self.batchNorm(self.relu(self.dropout(self.conv(x))))
        y = self.avgpool(y)
        return y


# class ClassificationLayer(Model):
class ClassificationLayer(layers.Layer):
    def __init__(self, num_classes):
        super().__init__()
        self.avgpool = layers.MaxPooling2D((7, 7), strides=7)
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(num_classes)
        self.softmax = layers.Activation("softmax")

    def call(self, x):
        y = self.avgpool(x)
        y = self.flatten(y)
        y = self.dense(y)
        y = self.softmax(y)


        return y

class DenseNet(Model):
    def __init__(self, config):
        super().__init__()
        self.growthRate = config["model"]["growth_rate"]
        self.compressionFactor = config["model"]["compression_factor"]
        self.num_classes = config["input"]["num_classes"]

        if config["model"]["name"] == "DenseNet121":
            numDenseBlocks = [6, 12, 24, 16]
            # numDenseBlocks = [1,1,1,1]
        elif config["model"]["name"] == "DenseNet169":
            numDenseBlocks = [6, 12, 32, 32]
        elif config["model"]["name"] == "DenseNet201":
            numDenseBlocks = [6, 12, 48, 32]
        elif config["model"]["name"] == "DenseNet201":
            numDenseBlocks = [6, 12, 64, 48]

        self.conv_init = layers.Conv2D(2 * self.growthRate, kernel_size=7, strides=2, padding="same")
        self.layers_init = [self.conv_init,
                            layers.Activation("relu"),
                            layers.MaxPooling2D((2, 2), strides=2)]


        ## denseblocks
        self.denseblock1 = DenseBlock(self.growthRate, numDenseBlocks[0])
        self.denseblock2 = DenseBlock(self.growthRate, numDenseBlocks[1])
        self.denseblock3 = DenseBlock(self.growthRate, numDenseBlocks[2])
        self.denseblock4 = DenseBlock(self.growthRate, numDenseBlocks[3])

        ## Classification layer
        self.classification = ClassificationLayer(self.num_classes)

    def call(self, x):
        # initial convolution and pooling
        # y = layers.Conv2D(2 * self.growthRate, kernel_size=7, strides=2, padding="same")(x)
        # y = layers.Activation("relu")(y)
        # y = layers.MaxPooling2D((2, 2), strides=2)(y) # paper uses (3,3) kernel but not consistent with output size


        # pdb.set_trace()
        y = x
        for layer in self.layers_init.layers:
            y = layer(y)


        y, numChannels = self.denseblock1(y)

        y = TransitionLayer(numChannels, self.compressionFactor)(y)

        y, numChannels = self.denseblock2(y)
        y = TransitionLayer(numChannels, self.compressionFactor)(y)


        y, numChannels = self.denseblock3(y)
        y = TransitionLayer(numChannels, self.compressionFactor)(y)

        y, numChannels = self.denseblock4(y)

        y = self.classification(y)

        return y


import json
if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)
    model = DenseNet(config)
    x = tf.random.uniform((2, 224,224, 3))
    y = model(x)
    print(y.shape)
    print(model.summary())

    # Testing denseblock (because has 0 parameters in model.summary())
    # model = DenseBlock(12, 6)
    # model.build((2,56,56,24))
    # print(model.summary())

    pdb.set_trace()
