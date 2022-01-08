# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 15:07:34 2022

@author: Restandsleep
"""
#importing libraries
import cv2 as cv
import os
import glob
import scipy.io
from mat4py import loadmat

a =  loadmat(r"C:\Users\Restandsleep\Desktop\VIT\Personal\hand_dataset\training_dataset\training_data\annotations\Buffy_10.mat")
print(a)
print("\n")
b = list(a.values())
print("\n")
print(b)
print("\n")
print((b[0][1]))
print("\n")
c = list(b[0][1].values())
print("\n")
print(c[1])


import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop,Adam
import tensorflow_io as tfio
import tensorflow_datasets as tfds



#Test space










#Importing data


#Setting directories
train_dir = r"C:\Users\Restandsleep\Desktop\VIT\Personal\hand_dataset\training_dataset\training_data\images"
train_dir = glob.glob(os.path.join(train_dir, "*.jpg"))

train_dir_val = r"C:\Users\Restandsleep\Desktop\VIT\Personal\hand_dataset\training_dataset\training_data\annotations"
train_dir_val = glob.glob(os.path.join(train_dir_val, "*.mat"))


validation_dir = r"C:\Users\Restandsleep\Desktop\VIT\Personal\hand_dataset\validation_dataset\validation_data\images"
validation_dir = glob.glob(os.path.join(validation_dir, "*.jpg"))

validation_dir_val = r"C:\Users\Restandsleep\Desktop\VIT\Personal\hand_dataset\validation_dataset\validation_data\annotations"
validation_dir_val = glob.glob(os.path.join(validation_dir_val, "*.mat"))


test_dir = r"C:\Users\Restandsleep\Desktop\VIT\Personal\hand_dataset\test_dataset\test_data\images"
test_dir = glob.glob(os.path.join(test_dir, "*.jpg"))

test_dir_val = r"C:\Users\Restandsleep\Desktop\VIT\Personal\hand_dataset\test_dataset\test_data\annotations"
test_dir_val = glob.glob(os.path.join(test_dir_val, "*.mat"))







#Importing images and labels



#mat  data process function
def getvalues(data):
    b = list(data.values())
    i = len(b[0])
    hands = i
    coor = []
    for x in range(0,i):
        c = list(b[0][x].values())
        

#inporting location data
train_data = []
test_data = []
validation_data = []

for filename in train_dir_val:
    data = loadmat(filename)
    train_data.append(data)
    
for filename in validation_dir_val:
    data = loadmat(filename)

for filename in test_dir_val:
    data = loadmat(filename)

train_ds = tf.data.Dataset.from_tensor_slices((train_dir,train_data))


for filename in train_dir:    
    img = cv.imread(filename)




# for i in train_data:
#     print(i , "\n")

    
    
    
    
# data = loadmat(r"C:\Users\Restandsleep\Desktop\VIT\Personal\hand_dataset\training_dataset\training_data\annotations\Buffy_2.mat")
    


