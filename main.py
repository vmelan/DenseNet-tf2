import json
import logging
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from data_loader import DataLoader
from model import DenseNet

import pdb

print(tf.__version__)


def main(config):
    # Load CIFAR data
    data = DataLoader(config)
    train_loader, test_loader = data.prepare_data()

    model = DenseNet(config)

    model.build((config["trainer"]["batch_size"], 224, 224, 3))
    print(model.summary())

    optimizer = tf.keras.optimizers.Adam(lr=0.001)
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    train_loss = tf.keras.metrics.Mean(name="loss", dtype=tf.float32)
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')


    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)


    for epoch in range(config["trainer"]["epochs"]):
        for step, (images, labels) in tqdm(enumerate(train_loader), total=int(data.get_len() / config["trainer"]["batch_size"])):
            train_step(images, labels)
        template = 'Epoch {}, Loss: {:.4f}, Accuracy: {:.4f}'
        print (template.format(epoch+1,
                             train_loss.result(),
                             train_accuracy.result()*100
                             )
             )
            # train_accuracy.reset_states()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s: %(message)s')

    with open("config.json", "r") as f:
        config = json.load(f)
    main(config)
