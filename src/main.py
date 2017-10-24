#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Generic classifier with multiple models
    Models -> (Xception, VGG16, VGG19, ResNet50, InceptionV3, MobileNet)

    Name: train.py
    Author: Gabriel Kirsten Menezes (gabriel.kirsten@hotmail.com)
    GitHub: https://github.com/gabrielkirsten/cnn_keras

"""

import time
import os
import argparse
import itertools
import matplotlib.pyplot as plt
import numpy as np
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress warnings

# =========================================================
# Constants and hyperparameters
# =========================================================

START_TIME = time.time()
IMG_WIDTH, IMG_HEIGHT = 256, 256
TRAIN_DATA_DIR = "../data/train"
VALIDATION_DATA_DIR = "../data/validation"
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.0001
CLASS_NAMES = ['ferrugemAsiatica', 'folhaSaudavel', 'fundo', 'manchaAlvo', 'mildio', 'oidio']

# =========================================================
# End of constants and hyperparameters
# =========================================================

def get_args():
    """Read the arguments of the program."""
    arg_parse = argparse.ArgumentParser()

    arg_parse.add_argument("-a", "--architecture", required=True,
                           help="Select architecture(Xception, VGG16, VGG19, ResNet50" +
                           ", InceptionV3, MobileNet)",
                           default=None, type=str)

    arg_parse.add_argument("-f", "--fineTuningRate", required=True,
                           help="Fine tunning rate", default=None, type=int)

    return vars(arg_parse.parse_args())


def plot_confusion_matrix(confusion_matrix_to_print, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
        This function prints applicationsand plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
    """

    plt.imshow(confusion_matrix_to_print, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = confusion_matrix_to_print.max() / 2.
    for i, j in itertools.product(range(confusion_matrix_to_print.shape[0]),
                                  range(confusion_matrix_to_print.shape[1])):
        plt.text(j, i, format(confusion_matrix_to_print[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if confusion_matrix_to_print[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def make_confusion_matrix_and_plot(validation_generator, file_name, model_final):
    """Predict and plot confusion matrix"""

    validation_features = model_final.predict_generator(validation_generator,
                                                        validation_generator.samples,
                                                        verbose=1)

    plt.figure()

    plot_confusion_matrix(confusion_matrix(np.argmax(validation_features, axis=1),
                                           validation_generator.classes),
                          classes=CLASS_NAMES,
                          title='Confusion matrix - ' + file_name)

    plt.savefig('../output_images/' + file_name + '.png')

    print("Total time after generate confusion matrix: %s" %
          (time.time() - START_TIME))


def main():
    """The main function"""

    args = get_args()  # read args

    if args["fineTuningRate"] != -1:
        if args["architecture"] == "Xception":
            model = applications.Xception(
                weights="imagenet", include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
        elif args["architecture"] == "VGG16":
            model = applications.VGG16(
                weights="imagenet", include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
        elif args["architecture"] == "VGG19":
            model = applications.VGG19(
                weights="imagenet", include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
        elif args["architecture"] == "ResNet50":
            model = applications.ResNet50(
                weights="imagenet", include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
        elif args["architecture"] == "InceptionV3":
            model = applications.InceptionV3(
                weights="imagenet", include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
        elif args["architecture"] == "MobileNet":
            model = applications.MobileNet(
                weights="imagenet", include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))

        for layer in model.layers[:int(len(model.layers) * (args["fineTuningRate"] / 100))]:
            layer.trainable = False

    else:  # without transfer learning
        if args["architecture"] == "Xception":
            model = applications.Xception(
                weights=None, include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
        elif args["architecture"] == "VGG16":
            model = applications.VGG16(
                weights=None, include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
        elif args["architecture"] == "VGG19":
            model = applications.VGG19(
                weights=None, include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
        elif args["architecture"] == "ResNet50":
            model = applications.ResNet50(
                weights=None, include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
        elif args["architecture"] == "InceptionV3":
            model = applications.InceptionV3(
                weights=None, include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
        elif args["architecture"] == "MobileNet":
            model = applications.MobileNet(
                weights=None, include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
        for layer in model.layers:
            layer.trainable = True

    # Adding custom Layers
    new_custom_layers = model.output
    new_custom_layers = Flatten()(new_custom_layers)
    new_custom_layers = Dense(1024, activation="relu")(new_custom_layers)
    new_custom_layers = Dropout(0.5)(new_custom_layers)
    new_custom_layers = Dense(1024, activation="relu")(new_custom_layers)
    predictions = Dense(6, activation="softmax")(new_custom_layers)

    # creating the final model
    model_final = Model(inputs=model.input, outputs=predictions)

    # compile the model
    model_final.compile(loss="categorical_crossentropy",
                        optimizer=optimizers.SGD(lr=LEARNING_RATE, momentum=0.9),
                        metrics=["accuracy"])

    # Initiate the train and test generators with data Augumentation
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True,
        fill_mode="nearest",
        zoom_range=0.3,
        width_shift_range=0.3,
        height_shift_range=0.3,
        rotation_range=30)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DATA_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        shuffle=True,
        class_mode="categorical")

    test_datagen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True,
        fill_mode="nearest",
        zoom_range=0.3,
        width_shift_range=0.3,
        height_shift_range=0.3,
        rotation_range=30)

    validation_generator = test_datagen.flow_from_directory(
        VALIDATION_DATA_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        shuffle=True,
        class_mode="categorical")

    # select .h5 filename
    if args["fineTuningRate"] == 100:
        file_name = args["architecture"] + \
            '_transfer_learning'
    elif args["fineTuningRate"] == -1:
        file_name = args["architecture"] + \
            '_without_transfer_learning'
    else:
        file_name = args["architecture"] + \
            '_fine_tunning_' + str(args["fineTuningRate"])

    # Save the model according to the conditions
    checkpoint = ModelCheckpoint("../models_checkpoints/" + file_name + ".h5", monitor='val_acc',
                                 verbose=1, save_best_only=True, save_weights_only=False,
                                 mode='auto', period=1)

    # Train the model
    model_final.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[checkpoint],
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE)

    print "Total time to train: %s" % (time.time() - START_TIME)

    validation_generator = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
        VALIDATION_DATA_DIR,
        batch_size=1,
        shuffle=False,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        class_mode="categorical")

    make_confusion_matrix_and_plot(
        validation_generator, file_name, model_final)


if __name__ == '__main__':
    main()
