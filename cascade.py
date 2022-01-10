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

import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop,Adam
import tensorflow_io as tfio

########################################################################################################################################

#Test space

# a =  loadmat(r"C:\Users\Restandsleep\Desktop\VIT\Personal\hand_dataset\training_dataset\training_data\annotations\Buffy_1.mat")
# print(a)
# print("\n")
# b = list(a.values())
# print("\n")
# print(b[0])
# print("\n")
# print((b[0][1]))
# print("\n")
# c = list(b[0].values())
# print("\n")
# print(c)


# a =  loadmat(r"C:\Users\Restandsleep\Desktop\VIT\Personal\hand_dataset\training_dataset\training_data\annotations\VOC2010_998.mat")
# print(a)

# b = list(a.values())
# print("\n")
# print(b[0])
# print("\n")
# print((b[0][1]))
# print("\n")
# c = list(b[0][1].values())
# print("\n")



########################################################################################################################################

#Importing data


#Setting directories
train_dir = r"C:\Users\Restandsleep\Desktop\VIT\Personal\hand_dataset\training_dataset\training_data\images"
train_dir = glob.glob(os.path.join(train_dir, "*.jpg"))

train_dir_val = r"C:\Users\Restandsleep\Desktop\VIT\Personal\hand_dataset\training_dataset\training_data\annotations"
train_dir_val = glob.glob(os.path.join(train_dir_val, "*.mat"))

train_dir_negative = r"C:\Users\Restandsleep\Desktop\VIT\Personal\hand_dataset\training_dataset\training_data\images_negative"
train_dir_negative = glob.glob(os.path.join(train_dir, "*.jpg"))


validation_dir = r"C:\Users\Restandsleep\Desktop\VIT\Personal\hand_dataset\validation_dataset\validation_data\images"
validation_dir = glob.glob(os.path.join(validation_dir, "*.jpg"))

validation_dir_val = r"C:\Users\Restandsleep\Desktop\VIT\Personal\hand_dataset\validation_dataset\validation_data\annotations"
validation_dir_val = glob.glob(os.path.join(validation_dir_val, "*.mat"))


test_dir = r"C:\Users\Restandsleep\Desktop\VIT\Personal\hand_dataset\test_dataset\test_data\images"
test_dir = glob.glob(os.path.join(test_dir, "*.jpg"))

test_dir_val = r"C:\Users\Restandsleep\Desktop\VIT\Personal\hand_dataset\test_dataset\test_data\annotations"
test_dir_val = glob.glob(os.path.join(test_dir_val, "*.mat"))







#Importing images and labels





#Image Data importing and processing


def getimage(dir):
    image  = []
    x = 0
    for filename in dir:
        img = cv.imread(filename)
        image.append(img)
        x = x+1
    return image , x
        
list3, x = getimage(train_dir)
list2 , neg_samples = getimage(train_dir_negative)
for i in list2:
    list3.append(i)
train_data = pd.DataFrame(list3 , columns =['Image'] )


list3 , x = getimage(validation_dir)
validation_data = pd.DataFrame(list3 , columns =['Image'] )


list3 , x = getimage(test_dir)
test_data = pd.DataFrame(list3 , columns =['Image'] )



# #mat  data process function


def getvalues(data):
    b = 0
    b = list(data.values())
    i = len(b[0])
    coor = []
    htype = []
    if type(b[0]) == list:
        for x in range(0,i):
            y = list(b[0][x].values())
            coor.append(y[0:4])
            # print(y , "\n")
            if(len(y)>4):
                #Encoding Hand type
                if(y[4] == 'L'):
                    y[4] = 1
                elif(y[4] == 'R'):
                    y[4] = 2
                else:
                    y[4]=0
                htype.append(y[4])   
            else:
                htype.append(0)
        nhands = i    
    else:
        y = list(b[0].values())
        coor.append(y[0:4])      
        if(len(y)>4):
            #Encoding Hand type
            if(y[4] == 'L'):
                y[4] = 1
            elif(y[4] == 'R'):
                y[4] = 2
            else:
                y[4]=0    
            htype.append(y[4])   
        else:
            htype.append(0)
        nhands = 1    
    return nhands , coor, htype



# #mat import function


def importval(dir):
    noofhands = []
    hand_coordinates = []
    hand_type = []
    for filename in dir:
        data = loadmat(filename)
        
        nhands , coor , htype = getvalues(data)
        noofhands.append(nhands)
        hand_coordinates.append(coor)
        hand_type.append(htype)
    return noofhands , hand_coordinates , hand_type

# #inporting location data


noofhands_glob , hand_coordinates_glob , hand_type_glob =   importval(train_dir_val)
for i in range(0,neg_samples):
    noofhands_glob.append(0) 
    hand_coordinates_glob.append([[[]]])
    hand_type_glob.append([])
    
list3 = zip(noofhands_glob,hand_coordinates_glob,hand_type_glob)

train_data_output = pd.DataFrame(list3, columns=('no of hands' , 'hand coordinates' , 'hand type')) 


noofhands_glob,hand_coordinates_glob,hand_type_glob =   importval(validation_dir_val)
list3 = zip(noofhands_glob,hand_coordinates_glob,hand_type_glob)
validation_data_output = pd.DataFrame(list3, columns=('no of hands' , 'hand coordinates' , 'hand type')) 


noofhands_glob,hand_coordinates_glob,hand_type_glob =   importval(test_dir_val)
list3 = zip(noofhands_glob,hand_coordinates_glob,hand_type_glob)
test_data_output = pd.DataFrame(list3, columns=('no of hands' , 'hand coordinates' , 'hand type')) 
    

#shuffling training data


train_data_combined = pd.concat([train_data,train_data_output])#combining train data
train_data_combined = train_data_combined.sample(frac=1).reset_index(drop=True) #shuffling train data
train_data = train_data_combined.iloc[:,0]#breaking train data
train_data_output = train_data_combined.iloc[:,1:]#breaking train data


# numeric_feature_names = ['age', 'thalach', 'trestbps',  'chol', 'oldpeak']
# numeric_features = df[numeric_feature_names]
# numeric_features.head()    
    
    
# data = loadmat(r"C:\Users\Restandsleep\Desktop\VIT\Personal\hand_dataset\training_dataset\training_data\annotations\Buffy_2.mat")
    


