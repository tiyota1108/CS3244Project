# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 14:20:05 2019

@author: eunic
"""

#IMPORTING LIBRARIES
#conda install tensorflow-eigen
#pip install tensorflow
#pip install keras
#pip install keras-tqdm
#pip install split-folders

from keras import applications

from keras.preprocessing.image import ImageDataGenerator

from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import (Dropout, Flatten, Dense, Conv2D, 
                          Activation, MaxPooling2D)

from sklearn.model_selection import train_test_split

from keras_tqdm import TQDMNotebookCallback

import os, glob
from tqdm import tqdm
import pandas as pd
from sklearn.utils import shuffle
import numpy as np
import shutil
from keras.optimizers import Adam
adam = Adam(lr=1)

import os
os.chdir("C:/Users/eunic/OneDrive/Desktop/CS3244 Project")
os.listdir()

#removing corrupt file images 
from PIL import Image
import sys
for file in os.listdir("train"):
    #print(file)
    for img_file in os.listdir("train/" + file):
        #print(img_file)
        try:
            im = Image.open("train/" + file + "/" + img_file)
        except OSError as err:
            os.remove("train/" + file + "/" + img_file)
            #print("OS error: {0}".format(err))
        
        
#splitting data set into train, val, test sets; RUN ONLY ONCE in your computer
import split_folders
split_folders.ratio('train', output = "output", seed = 1337, ratio = (.8, .1, .1)) # default values

#setting values and directories
img_width, img_height = 256, 256 #change according to size of images

train_data_dir = 'output/train/'
validation_data_dir = 'output/val/'
test_data_dir = 'output/test/'

batch_size = 32

#IMPORTING IMAGES DATA
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip=False)

test_datagen = ImageDataGenerator(rescale=1. / 255) # this is the augmentation configuration we will use for testing: only rescaling

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical', shuffle=True)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')


#BUILDING THE CNN MODEL + FITTING THE DATA USING TRAIN AND VAL DATA
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(256, 256, 3)))
model.add(Activation('relu')) #tanh
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu')) #tanh
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(96))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10)) # 10 types of ships
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#TRYING CNN MODEL FOR SHIP CLASSIFICATION ON KERAS
from time import time
start = time()

history = model.fit_generator(
            train_generator,
            steps_per_epoch = 100, # give me more data
            epochs = 100, # The more the better, i.e. accuracy increases with the number of epochs, but will take a longer training time 
            validation_data = validation_generator,
            validation_steps = 100)

end = time()

hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print("Total Training Time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
# Total Training Time: 17:12:08.15

model.save_weights('sixth_try.h5')

import json
with open('trainHistory6.json', 'w') as history_file:
    json.dump(history.history, history_file)
    
#Save the model so that you don't have to fit_generator all the time
from keras.models import load_model
model.save('ship_model6.h5')
ship_model = load_model('ship_model6.h5')

#Accuracy of model's prediction on test data
loss, acc = ship_model.evaluate_generator(test_generator, steps=3, verbose=0) # to get the accuracy score
print(acc)
# accuracy is 0.5882353011299583

# summarize history for accuracy
import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy by Epoch')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()
plt.savefig('model6_acc.png')
plt.savefig('model6_acc.pdf')

# summarize history for loss
plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss by Epoch')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()
plt.savefig('model6_loss.png')
plt.savefig('model6_loss.pdf')

# save visualization of network architecture
#!pip install pydot
#!pip install graphviz
#!pip install pydotplus

from keras.utils.vis_utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

#Visualize Model
plot_model(ship_model, to_file='ship_model6.png', show_layer_names=True, show_shapes=True)

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sn

target_names = ["Container",
                "FireFightingVessel",
                "HeavyLoadCarrier",
                "Passenger",
                "Platform",
                "Reefer",
                "ReplenishmentVessel",
                "SupplyVessel",
                "TrainingVessel",
                "Tug"]

# Test set
test_dir = "C:/Users/eunic/OneDrive/Desktop/CS3244 Project/output/test"
num_of_test_samples = sum([len(files) for r, d, files in os.walk(test_dir)])

#Confution Matrix and Classification Report
Y_pred = ship_model.predict_generator(test_generator, num_of_test_samples//batch_size+1)
Y_pred = np.argmax(Y_pred, axis=1)

# Check that both are of equal size
# test_generator.classes.shape
# y_pred.shape

print('Confusion Matrix')
test_cm = confusion_matrix(test_generator.classes, y_pred)
#df_test_cm = pd.DataFrame(test_cm, range(10), range(10))
df_test_cm = pd.DataFrame(test_cm, target_names, target_names)
sn.set(font_scale=1.4) #for label size
sn.heatmap(df_test_cm, annot=True, annot_kws={"size": 16}) # font size

print(classification_report(test_generator.classes, y_pred, target_names=target_names))