import tensorflow as tf
import keras as k
from keras import layers

from keras_preprocessing.image import ImageDataGenerator
from keras.models import clone_model
from keras.applications import ResNet50

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

import cv2
import os
#import shutil

from PIL import Image


#classification

def classi(input_shape):
    inputs = layers.Input(shape=input_shape)
    #vgg19 = k.applications.VGG19(include_top=False, weights="imagenet", input_tensor=inputs)
    #x = vgg19(inputs, training=False)
    densenet121 = k.applications.DenseNet121(include_top=False, input_tensor=inputs)
    x = densenet121(inputs, training=True)
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.Activation("relu")(x)
    x = layers.BatchNormalization()(x)
    #classi layers
    for filters in [96, 128]:#, 256]:#, 320]:#, 512]:#, 1024, 2048]:
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.Activation("relu")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.Activation("relu")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPool2D(3, strides=2, padding="same")(x)

    #output
    x = layers.Dropout(rate=0.2)(x)
    x = layers.Flatten()(x)
    #x = layers.Dense(128, activation="relu")(x)
    output = layers.Dense(7, activation="softmax")(x)

    model = k.Model(inputs=inputs, outputs=output, name="classification")
    return model

classification = classi((256,256,3))
classification.summary()

#classification.save("classification.keras")


cls_train_gt = 'train_ground_truth/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv'#("classi/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv")
cls_val = r'validation/ISIC2018_Task3_Validation_Input/'
cls_val_gt = "validation_ground_truth/ISIC2018_Task3_Validation_GroundTruth/ISIC2018_Task3_Validation_GroundTruth.csv"

#weighted binary loss
def get_weights(labels):
    cols = len(labels.columns)-2 #assumes 1 column for image ids
    weights = {}
    for i in range(cols+1):
        weights[i] = 1-np.mean(labels[labels.columns[i+1]].tolist())
    return weights

print(get_weights(pd.read_csv(cls_train_gt)))

from keras import backend as K
def f_score(y_true, y_pred, threshold=0.1, beta=2):
    tp = tp_score(y_true, y_pred, threshold)
    fp = fp_score(y_true, y_pred, threshold)
    fn = fn_score(y_true, y_pred, threshold)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return (1+beta**2) * ((precision * recall) / ((beta**2)*precision + recall))

def tp_score(y_true, y_pred, threshold=0.1):
    tp_3d = K.concatenate(
        [
            K.cast(K.expand_dims(K.flatten(y_true)), 'bool'),
            K.cast(K.expand_dims(K.flatten(K.greater(y_pred, K.constant(threshold)))), 'bool'),
            K.cast(K.ones_like(K.expand_dims(K.flatten(y_pred))), 'bool')
        ], axis=1
    )
    tp = K.sum(K.cast(K.all(tp_3d, axis=1), 'int32'))
    return tp

def fp_score(y_true, y_pred, threshold=0.1):
    fp_3d = K.concatenate(
        [
            K.cast(K.expand_dims(K.flatten(K.abs(y_true - K.ones_like(y_true)))), 'bool'),
            K.cast(K.expand_dims(K.flatten(K.greater(y_pred, K.constant(threshold)))), 'bool'),
            K.cast(K.ones_like(K.expand_dims(K.flatten(y_pred))), 'bool')
        ], axis=-1
    )

    fp = K.sum(K.cast(K.all(fp_3d, axis=1), 'int32'))
    return fp

def fn_score(y_true, y_pred, threshold=0.1):
    fn_3d = K.concatenate(
        [
            K.cast(K.expand_dims(K.flatten(y_true)), 'bool'),
            K.cast(K.expand_dims(K.flatten(K.abs(K.cast(K.greater(y_pred, K.constant(threshold)), 'float') - K.ones_like(y_pred)))), 'bool'),
            K.cast(K.ones_like(K.expand_dims(K.flatten(y_pred))), 'bool')
        ], axis=1
    )

    fn = K.sum(K.cast(K.all(fn_3d, axis=1), 'int32'))
    return fn

def precision_score(y_true, y_pred, threshold=0.1):
    tp = tp_score(y_true, y_pred, threshold)
    fp = fp_score(y_true, y_pred, threshold)
    return tp / (tp + fp)

def recall_score(y_true, y_pred, threshold=0.1):
    tp = tp_score(y_true, y_pred, threshold)
    fn = fn_score(y_true, y_pred, threshold)
    return tp / (tp + fn)

##########################################################

cls_datagen = ImageDataGenerator(rotation_range=0,
                                horizontal_flip=True,
                                vertical_flip=True,
                                width_shift_range=0.15,
                                height_shift_range=0.15
                                )

##############################################################################

# For loading classification labels and images.
# Standard Function, random sampling
def load_images_and_labels(images_path, labels_path, batch_size, image_shape, verbose=False):
    ds_images = []
    ds_labels = []
    data_indexes = []
    labels = pd.read_csv(labels_path)
    images = os.listdir(images_path)
    if verbose:
        print(f"loading images from {images_path} and labels from {labels_path}")
    for i in range(batch_size):
        random_index = np.random.randint(0, len(images)-2)
        if random_index >= len(images):
            random_index -=1
        img = cv2.imread(os.path.join(images_path, images[random_index]))
        row = labels.iloc[random_index, 1:]

        if img is not None and row is not None:
            if random_index not in data_indexes:
                data_indexes.append(random_index)
                ds_images.append(np.array(cv2.resize(img, dsize=image_shape)))
                ds_labels.append(row.values)
    return np.array(ds_images).astype(np.int16), np.array(ds_labels).astype(np.int16)

# Equal Sampling
def load_images_and_labels_equal(images_path, labels_path, batch_size, num_classes, image_shape, verbose=False):
    ds_images = []
    ds_labels = []
    samples_per_class = []
    class_indexes = []
    labels = pd.read_csv(labels_path)
    class_names = np.array(labels.columns[1:])
    images = os.listdir(images_path)
    for i in range(num_classes):
        samples_per_class.append(batch_size//num_classes)
    for i in range(batch_size%num_classes):
        samples_per_class[i] += 1
    instances_per_class = labels.sum(axis=0, numeric_only=True)
    if verbose:
        print(f"loading images from {images_path} and labels from {labels_path} with equal sampling")
    for i in range(num_classes):
        #get all row indexes with 1 in ith row
        class_indexes = []
        p = 0
        for x in labels.iloc:
            if x[i+1] == 1:
                class_indexes.append(p)
            p+=1
        print(f"samples for {i} class: {samples_per_class[i]}, class_indexes len: {len(class_indexes)}")
        for x in range(samples_per_class[i]):
            random_index = np.random.randint(0, instances_per_class[i])
            ind = class_indexes[random_index]
            img = cv2.imread(os.path.join(images_path, images[ind]))
            row = labels.iloc[ind, 1:]
            if img is not None and row is not None:
                if not (samples_per_class[i] >= len(class_indexes)-1):               
                    class_indexes.pop(random_index)
                    instances_per_class[i] -= 1
                ds_images.append(np.array(cv2.resize(img, dsize=image_shape), dtype="uint8"))
                #assuming 1 column for image id
                #ds_labels.append(row.values) #for array labels
                ds_labels.append(class_names[np.where(row.values == 1)[0][0]]) #for string labels - make sure to remove .astype for labels
                #ds_labels.append(np.where(row.values == 1)[0][0]) #for integer labels
        print("1 class loaded")
        print(f"len ds_images: {len(ds_images)}")
        print(f"len ds_labels: {len(ds_labels)}")
    return np.array(ds_images).astype(np.uint8), pd.get_dummies(ds_labels).to_numpy()#k.utils.to_categorical(ds_labels, num_classes=num_classes)


##############################################


#classification = k.models.load_model('classification.keras') #reset weights
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
classification.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy", f_score, precision_score, 'AUC']) 

callback_list = []#[tf.keras.callbacks.EarlyStopping(patience=1.5)] #can adjust to improve accuracy
"""
batch_size=32
spe = 2 #steps per epoch
epochs = 50 # set to 1 for debugging purposes
"""

seed = 123

cls_val = r'validation/ISIC2018_Task3_Validation_Input/'
cls_val_gt = "validation_ground_truth/ISIC2018_Task3_Validation_GroundTruth/ISIC2018_Task3_Validation_GroundTruth.csv"

cls_train = r'train/ISIC2018_Task3_Training_Input/'#r"classi/ISIC2018_Task3_Training_Input/ISIC2018_Task3_Training_Input/" 
cls_train_gt = 'train_ground_truth/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv'#("classi/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv")
cls_train_gt_ham10000 = 'ham10000/HAM10000_metadata.csv'

epochs=60
batch_size = 2
spe=16
"""
length=len(pd.read_csv(cls_train_gt))//64
b_max= 60 # set this based on  how much your  memory can hold
batch_size=sorted([int(length/n) for n in range(1,length+1) if length % n ==0 and 
                  length/n<=b_max],reverse=True)[0] 
spe=int(length/batch_size)
"""


print(batch_size*spe*epochs)
df_train= pd.read_csv(cls_train_gt_ham10000)
#df_validation = pd.read_csv
train_ds, train_gt = load_images_and_labels_equal(cls_train, cls_train_gt, batch_size*spe*epochs, 7, (256,256), True)
val_ds, val_gt = load_images_and_labels_equal(cls_val, cls_val_gt, batch_size*spe*epochs//2, 7, (256,256), True)
print(train_gt)
#train_data = cls_datagen.flow_from_dataframe(dataframe=df_train, directory=cls_train, x_col="image_id", y_col="dx", target_size=(256,256), batch_size=epochs*batch_size*spe, shuffle=True)
#validation_data = cls_datagen.flow_from_dataframe(dataframe=df, directory=cls_train, x_col="image_id", y_col="dx", target_size=(256,256), batch_size=epochs*batch_size*spe, shuffle=True)
#print(f"train_ds len: {len(train_ds)}, train labels len: {len(train_gt)}")
#print(f"val_ds len: {len(val_ds)}, val labels len: {len(val_gt)}")
cls_train_gen = cls_datagen.flow(x=train_ds, y=train_gt, seed=seed, batch_size=batch_size*spe, shuffle=False)
val_train_gen = cls_datagen.flow(x=val_ds, y=val_gt, seed=seed, batch_size=batch_size*spe//2, shuffle=False)

print(f"--------------- Start training -----------------")

history = classification.fit(cls_train_gen, steps_per_epoch=len(cls_train_gen.x)//cls_train_gen.batch_size, epochs=epochs,
                                #class_weight=get_weights(pd.read_csv(cls_train_gt)), 
                                batch_size=batch_size, callbacks=callback_list, verbose=2,
                                validation_data=(val_train_gen.x, val_train_gen.y), validation_steps=len(val_train_gen.x)//val_train_gen.batch_size)

print(f"--------------- Done training -----------------")

classification.save_weights("classification_final_0.h5")
