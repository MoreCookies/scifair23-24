
#import libraries
import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

#do data manipulation 

#placeholder
model = k.Sequential([
    layers.Conv2D(64, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])
#https://www.geeksforgeeks.org/u-net-architecture-explained/ --> unet structure, easy to implement
#functions to add variation to data
# preprocessing.RandomContrast(factor=0.5),
#preprocessing.RandomFlip(mode='horizontal'), # meaning, left-to-right
# preprocessing.RandomFlip(mode='vertical'), # meaning, top-to-bottom
# preprocessing.RandomWidth(factor=0.15), # horizontal stretch
# preprocessing.RandomRotation(factor=0.20),
# preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),

#layers.InputLayer(input_shape=[]), specify imput layer

"""
standard block format, add more blocks w/ increasing by power of 2 filters

layers.BatchNomralization(renorm=True),
layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'), #kernel filter should be "glorot_uniform" not sure tho, can specify if u want
layers.MaxPool2D(),

""" # or try to create U-Net structure
    


#compilation code here
optimizer = tf.k.optimizers.Adam(epsilon=0.01) #use adam

"""
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=["binary_accuracy"]) # we can actually use metrics without matplotlib!!!! woww!!! i think
"""