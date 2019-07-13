import tensorflow as tf
from tensorflow.keras import Model, layers
import numpy as np

class DenseBlock(layers.Layer):
    def __init__(self, growthRate, numDenseBlocks):
        super().__init__()

        self.listLayers = []
        for _ in range(numDenseBlocks):
            self.listLayers.append(BottleneckLayer(growthRate))

    def call(self, x):
        for layer in self.listLayers.layers:
            x = layer(x)
        return x

class BottleneckLayer(layers.Layer):
    def __init__(self, growthRate):
        super().__init__()
        self.conv1 = layers.Conv2D(4 * growthRate, kernel_size=1, strides=1, padding="same")
        self.conv2 = layers.Conv2D(growthRate, kernel_size=3, strides=1, padding="same")
        self.bn1 = layers.BatchNormalization(momentum=0.99, epsilon=0.001)
        self.bn2 = layers.BatchNormalization(momentum=0.99, epsilon=0.001)

        self.listLayers = [self.conv1,
                           layers.Activation("relu"),
                           self.bn1,
                           self.conv2,
                           layers.Activation("relu"),
                           self.bn2]

    def call(self, x):
        y = x
        for layer in self.listLayers.layers:
            y = layer(y)
        y = layers.concatenate([x, y])
        return y


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
        y = self.batchNorm(self.relu(self.dropout(self.conv(x))))
        y = self.avgpool(y)
        return y


class ClassificationLayer(layers.Layer):
    def __init__(self, num_classes):
        super().__init__()
        self.avgpool = layers.GlobalAveragePooling2D()
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
        growthRate = config["model"]["growth_rate"]
        compressionFactor = config["model"]["compression_factor"]
        num_classes = config["input"]["num_classes"]

        if config["model"]["name"] == "DenseNet121":
            numDenseBlocks = [6, 12, 24, 16]
        elif config["model"]["name"] == "DenseNet169":
            numDenseBlocks = [6, 12, 32, 32]
        elif config["model"]["name"] == "DenseNet201":
            numDenseBlocks = [6, 12, 48, 32]
        elif config["model"]["name"] == "DenseNet201":
            numDenseBlocks = [6, 12, 64, 48]

        # initial convolution and pooling
        self.conv_init = layers.Conv2D(2 * growthRate, kernel_size=7, strides=2, padding="same")
        self.layers_init = [self.conv_init,
                            layers.Activation("relu"),
                            layers.MaxPooling2D((2, 2), strides=2)]

        numChannels = 2 * growthRate

        ## denseblocks and transition layers
        self.denseblock1 = DenseBlock(growthRate, numDenseBlocks[0])
        numChannels += numDenseBlocks[0] * growthRate
        self.transition1 = TransitionLayer(numChannels, compressionFactor)
        self.denseblock2 = DenseBlock(growthRate, numDenseBlocks[1])
        numChannels += numDenseBlocks[1] * growthRate
        self.transition2 = TransitionLayer(numChannels, compressionFactor)
        self.denseblock3 = DenseBlock(growthRate, numDenseBlocks[2])
        numChannels += numDenseBlocks[2] * growthRate
        self.transition3 = TransitionLayer(numChannels, compressionFactor)
        self.denseblock4 = DenseBlock(growthRate, numDenseBlocks[3])

        ## Classification layer
        self.classification = ClassificationLayer(num_classes)


    def call(self, x):
        y = x
        for layer in self.layers_init.layers:
            y = layer(y)
        y = self.denseblock1(y)
        y = self.transition1(y)
        y = self.denseblock2(y)
        y = self.transition2(y)
        y = self.denseblock3(y)
        y = self.transition3(y)
        y = self.denseblock4(y)
        y = self.classification(y)

        return y


if __name__ == "__main__":
    import json
    with open("config.json", "r") as f:
        config = json.load(f)
    model = DenseNet(config)
    x = tf.random.uniform((2, 224,224, 3))
    y = model(x)
    print(y.shape)
    print(model.summary())
