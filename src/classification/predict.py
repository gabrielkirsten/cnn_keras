#!/usr/bin/python
# -*- coding: utf-8 -*-

"""

    Abstract class for classifiers.

    Name: classifier.py
    Author: Gabriel Kirsten Menezes (gabriel.kirsten@hotmail.com)

"""

import numpy
import cv2
import argparse

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Flatten, Dense
import sys

img_width, img_height = 28, 28


def get_args():
    """Read the arguments of program."""
    ap = argparse.ArgumentParser()

    ap.add_argument("-cl", "--classes", required=True, help="Classes names",
                    default=None, type=str)

    ap.add_argument("-i", "--inputimage", required=True, help="Input image " +
                    "for the classifier", default=None, type=str)
    
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


# read args
get_args()

# load classes
outputClasses = args["classes"].split()

# load image
img = cv2.imread(args["inputimage"])
img = cv2.resize(img, (img_width, img_height))

model = create_model(args["model"])

model.load_weights('./weith_neuralnet.h5')

arr = numpy.array(img).reshape((img_width, img_height, 3))
arr = numpy.expand_dims(arr, axis=0)
prediction = model.predict(arr)[0]

# name of the best class
bestclass = ''

bestconf = -1

for n in [0, 1]:
    if (prediction[n] > bestconf):
        bestclass = str(n)
        bestconf = prediction[n]

print 'Class: ' + outputClasses[int(bestclass)] + '(' + bestclass + ')'
+ ' with ' + str(bestconf * 100) + '% confidence.'
