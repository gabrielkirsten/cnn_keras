#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import os.path
import tensorflow

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Flatten, Dense

img_width, img_height = 28, 28

train_data_dir = '../../data/train'

validation_data_dir = '../../data/validation'

train_samples = 0
for directory_class in [x[0] for x in os.walk(train_data_dir)]:
    train_samples = train_samples
    + len([name for name in os.listdir(directory_class)
          if os.path.isfile(os.path.join(directory_class, name))])

validation_samples = 0
for directory_class in [x[0] for x in os.walk(train_data_dir)]:
    validation_samples = validation_samples
    + len([name for name in os.listdir(directory_class)
          if os.path.isfile(os.path.join(directory_class, name))])

epoch = 6


def get_args():
    """Read the arguments of program."""
    ap = argparse.ArgumentParser()

    ap.add_argument("-m", "--model", required=True, help="CNN Model",
                    default=None, type=str)

    return vars(ap.parse_args())


def create_model(opc):

    if (opc == 1):
        """Create the custom model"""
        model = Sequential()
        model.add(Convolution2D(16, (5, 5),
                                activation='relu',
                                input_shape=(img_width, img_height, 3)))
        model.add(MaxPooling2D(2, 2))

        model.add(Convolution2D(32, (5, 5), activation='relu'))
        model.add(MaxPooling2D(2, 2))

        model.add(Flatten())
        model.add(Dense(1000, activation='relu'))

        model.add(Dense(8, activation='softmax'))

    elif (opc == 2):
        """Create GooLeNet - Inception module (2014)"""
        model = Graph()
        model.add_input(name='n00', input_shape=(1, 28, 28))

        # layer 1
        model.add_node(Convolution2D(64, 1, 1, activation='relu'),
                       name='n11', input='n00')
        model.add_node(Flatten(), name='n11_f', input='n11')

        model.add_node(Convolution2D(96, 1, 1, activation='relu'),
                       name='n12', input='n00')

        model.add_node(Convolution2D(16, 1, 1, activation='relu'),
                       name='n13', input='n00')

        model.add_node(MaxPooling2D((3, 3), strides=(2, 2)),
                       name='n14', input='n00')

        # layer 2
        model.add_node(Convolution2D(128, 3, 3, activation='relu'),
                       name='n22', input='n12')
        model.add_node(Flatten(), name='n22_f', input='n22')

        model.add_node(Convolution2D(32, 5, 5, activation='relu'),
                       name='n23', input='n13')
        model.add_node(Flatten(), name='n23_f', input='n23')

        model.add_node(Convolution2D(32, 1, 1, activation='relu'),
                       name='n24', input='n14')
        model.add_node(Flatten(), name='n24_f', input='n24')

        # output layer
        model.add_node(Dense(1024, activation='relu'), name='layer4',
                       inputs=['n11_f', 'n22_f', 'n23_f', 'n24_f'],
                       merge_mode='concat')
        model.add_node(Dense(10, activation='softmax'), name='layer5',
                       input='layer4')
        model.add_output(name='output1', input='layer5')

    return model


def compile_model(model):
    """Compile the model"""
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model

# read args
get_args()

model = create_model(args["model"])
model = compile_model(model)

test_datagen = ImageDataGenerator(rescale=1./255)

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='categorical')

model.fit_generator(
        train_generator,
        samples_per_epoch=train_samples,
        nb_epoch=epoch,
        validation_data=validation_generator,
        nb_val_samples=validation_samples)

model.save_weights('../../data/weith_neuralnet.h5')
