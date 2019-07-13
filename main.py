import json
import logging

import tensorflow as tf
from tensorflow import keras

from data_loader import DataLoader

import pdb
import numpy as np

from tqdm import tqdm
from model import *


import matplotlib.pyplot as plt

print(tf.__version__)

def main(config):
    # Load CIFAR data
    data = DataLoader(config)
    X_train, y_train, X_test, y_test = data.load_cifar10(config["input"]["data_path"])
    train_loader, test_loader = data.prepare_data()
    # pdb.set_trace()

    # model = keras.applications.DenseNet121() # default input size for model : 224 x 224
    model = DenseNet(config)
    # model = LeNet(config)

    model.build((config["trainer"]["batch_size"], 224, 224, 3))
    print(model.summary())

    optimizer = tf.keras.optimizers.Adam(lr=0.001)


    loss_object = tf.keras.losses.CategoricalCrossentropy()
    train_loss = tf.keras.metrics.Mean(name="loss", dtype=tf.float32)
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')


    def train_step(images, labels):
        with tf.GradientTape() as tape:
            # pdb.set_trace()
            predictions = model(images)
            loss = loss_object(labels, predictions)
            # print("loss: {}".format(loss))
        gradients = tape.gradient(loss, model.trainable_variables)
        # gradients = [tf.clip_by_norm(g, 15) for g in gradients]
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    EPOCHS = 5

    for epoch in range(EPOCHS):
        # for step, (images, labels) in tqdm(enumerate(train_loader), total=int(len(X_train) / config["trainer"]["batch_size"])):
        for step, (images, labels) in enumerate(train_loader):
            train_step(images, labels)
            template = 'Step {}, Loss: {:.4f}, Accuracy: {:.4f}'
            print (template.format(step+1,
                                 train_loss.result(),
                                 train_accuracy.result()*100
                                 )
                 )
            # pdb.set_trace()
            # train_accuracy.reset_states()


    ## Keras api
    # tbCallback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
    # # tbCallback = None
    # model.compile(optimizer=optimizer, loss="categorical_crossentropy", callbacks=[tbCallback])
    #
    # # pdb.set_trace()
    # model.fit(train_loader, epochs=EPOCHS, verbose=2)




## Test lenet architecture
from LeNet import LeNet
def test_lenet(config):
    # Load CIFAR data
    data = DataLoader(config)
    X_train, y_train, X_test, y_test = data.load_cifar10(config["input"]["data_path"])
    train_loader, test_loader = data.prepare_data()
    # pdb.set_trace()

    # model = keras.applications.DenseNet121() # default input size for model : 224 x 224

    model = LeNet(config)

    avg_loss = tf.keras.metrics.Mean(name="loss", dtype=tf.float32)

    optimizer = tf.keras.optimizers.Adam(lr=0.001)
    for step, (images, labels) in enumerate(train_loader):
        with tf.GradientTape() as tape:
            preds = model(images)
            loss = tf.keras.backend.categorical_crossentropy(labels, preds) # y_true, y_pred
            avg_loss.update_state(loss)

        grads = tape.gradient(loss, model.trainable_variables)
        # clip gradient
        # grads = [tf.clip_by_norm(g, 15) for g in grads]
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        # print("loss:", loss)
        print("loss:", avg_loss.result().numpy())
        # print("grads:", grads[-1])

        ## Clear accumulated values
        avg_loss.reset_states()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s: %(message)s')

    with open("config.json", "r") as f:
        config = json.load(f)
    main(config)
    # test_lenet(config)
