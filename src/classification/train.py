#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import os.path

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


def create_model():
    """Create the model"""
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

    return model


def compile_model(model):
    """Compile the model"""
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model


model = create_model()
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
