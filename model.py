#import libraries
import numby as np
import pandas as pd

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

#do data manipulation 
df1 = x
df2 = y
df1.merge(df2, how = 'a', on = ['b', 'c'])
    #Renaming Series for merging
df1.rename(
    columns={"OLD COLUMN NAME": "NEW COLUMN NAME", "OLD COLUMN NAME 2": "NEW COLUMN NAME 2"},
    inplace=True,
)

#placeholder
#PLAN
#create u-net architecture --> segmented input map
#segmented input map --> layers.Dense() --> turn into classification task --> return binary label

#U-NET architecture
#input layer
foo = 64
data = None #set data to this value
class_num = data.shape()-2 #number of unique labels in data



model = layers.Conv2D(foo, 3, padding="same")(data)
model = layers.BatchNormalization()(model) #(x) --> using previous layer as input for next layer
model = layers.Activation("relu")(model)

for filter in [64,128,256, 512, 1028]:
    model = layers.Conv2D(filter, 3, padding="same",activation="relu")(model) # (filters, kernel_size, padding, activation)
    model = layers.BatchNormalization()(model)

    model = layers.Conv2D(filter, 3, padding="same",activation="relu")(model)
    model = layers.BatchNormalization()(model)

    model = layers.MaxPooling2D(3, strides=2, padding="same")(model) # (pool_size, strides, padding)

#expanding u-net

for filter in [1028, 512, 256, 128, 64]:
    model = layers.Conv2DTranspose(filter, 3, padding="same",activation="relu")(model)
    model = layers.BatchNormalization()(model)

    model = layers.Conv2DTranspose(filter, 3, padding="same",activation="relu")(model)
    model = layers.BatchNormalization()(model)

    model =  layers.UpSampling2D(2)(model)

classification = layers.Conv2D(class_num, 3, activation="softmax", padding="same")(x)

model_final = k.Model(data, classification)

""" 
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
"""
standard block format, add more blocks w/ increasing by power of 2 filters

layers.BatchNormalization(renorm=True),
layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'), #kernel filter should be "glorot_uniform" not sure tho, can specify if u want
layers.MaxPool2D(),

""" # or try to create U-Net structure
    


#compilation code here
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=["binary_accuracy"]) # we can actually use metrics without matplotlib!!!! woww!!! i think