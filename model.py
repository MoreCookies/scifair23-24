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
train_ds, test_ds = tf.keras.utils.split_dataset(data, left_size=0.8) #SPLIT DATASET --> COMBINE ALL DATA INTO ONE VARIABLE AND THIS WILL SPLIT IT
class_num = data.shape()-2 #number of unique labels in data

model = layers.InputLayer(input_shape=[256,256]) #change later

#data augmentation
model = preprocessing.RandomContrast(factor=0.1)(model)
model = preprocessing.RandomFlip()(model) #horizontal and vertical flipping
model = preprocessing.RandomRotation(factor=0.1)(model)

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

model_final = tf.keras.Model(data, classification)


#compilation code here
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=["binary_accuracy"]) # we can actually use metrics without matplotlib!!!! woww!!! i think

callback_list = [tf.keras.callbacks.EarlyStopping(patience=2)] #can adjust to improve accuracy

#early stopping used to prevent overfitting

model.fit(train_ds, epochs=80, callbacks=callback_list)