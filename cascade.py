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
from keras_preprocessing.image import ImageDataGenerator
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
train_dir_negative = glob.glob(os.path.join(train_dir_negative, "*.jpg"))


validation_dir = r"C:\Users\Restandsleep\Desktop\VIT\Personal\hand_dataset\validation_dataset\validation_data\images"
validation_dir = glob.glob(os.path.join(validation_dir, "*.jpg"))

validation_dir_val = r"C:\Users\Restandsleep\Desktop\VIT\Personal\hand_dataset\validation_dataset\validation_data\annotations"
validation_dir_val = glob.glob(os.path.join(validation_dir_val, "*.mat"))


test_dir = r"C:\Users\Restandsleep\Desktop\VIT\Personal\hand_dataset\test_dataset\test_data\images"
test_dir = glob.glob(os.path.join(test_dir, "*.jpg"))

test_dir_val = r"C:\Users\Restandsleep\Desktop\VIT\Personal\hand_dataset\test_dataset\test_data\annotations"
test_dir_val = glob.glob(os.path.join(test_dir_val, "*.mat"))







#Importing images and label





#Image Data importing and processing


def getimage(dir):
    # image  = []
  
    # for filename in dir:
    #     img = cv.imread(filename)
    #     image.append(img)
    return dir
        
        
        

train_dir = train_dir + train_dir_negative
neg_samples = len(train_dir_negative)
train_data = getimage(train_dir)
validation_data = getimage(validation_dir)
# test_data = getimage(test_dir)


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
        nhands = (i)   
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
        nhands = (1)   
    return nhands, coor, htype



# #mat import function


def importval(dir):
    noofhands = []
    hand_coordinates = []
    hand_type = []
    i = 0
    for filename in dir:
        data = loadmat(filename)
        i = i + 1
       
        nhands , coor , htype = getvalues(data)
        noofhands.append(nhands)
        hand_coordinates.append(coor)
        hand_type.append(htype)
    return noofhands , hand_coordinates , hand_type

#inporting location data

#importing train data
noofhands_glob , hand_coordinates_glob , hand_type_glob =   importval(train_dir_val)
for i in range(0,neg_samples):
    noofhands_glob.append(0) 
    hand_coordinates_glob.append([[0]])
    hand_type_glob.append([0])
    
list3 = zip(train_dir,noofhands_glob,hand_coordinates_glob,hand_type_glob)
#making train data dataframe
train_data_combined = pd.DataFrame(list3, columns=('filename' , 'no of hands' , 'hand coordinates' , 'class')) 
train_data_combined = train_data_combined.sample(frac=1).reset_index(drop=True) #shuffling train data

#importing validation data
noofhands_glob,hand_coordinates_glob,hand_type_glob =   importval(validation_dir_val)
#making validation data dataframe
list3 = zip(validation_dir,noofhands_glob,hand_coordinates_glob,hand_type_glob)
validation_data_combined = pd.DataFrame(list3, columns=('filename' ,'no of hands' , 'hand coordinates' , 'class')) 

#importing test data
# noofhands_glob,hand_coordinates_glob,hand_type_glob =   importval(test_dir_val)
#making test data dataframe
# list3 = zip(test_data,noofhands_glob,hand_coordinates_glob,hand_type_glob)
# test_data_combined = pd.DataFrame(list3, columns=('Image' ,'no of hands' , 'hand coordinates' , 'hand type')) 



train_data = train_data_combined.iloc[:,0]#breaking train data
# train_data = train_data.map(lambda x:cv.imread(x))
train_data_output = train_data_combined.iloc[:,1:]#breaking train data

validation_data = validation_data_combined.iloc[:,0]#breaking validation data
# validation_data = validation_data.map(lambda x:cv.imread(x))
validation_data_output = validation_data_combined.iloc[:,1:]#breaking validation data


col_list = ['no of hands' ]
for x in col_list:
    train_data_combined[x] = train_data_combined[x].map(lambda x: np.asarray(x).astype(np.float32))
    validation_data_combined[x] = validation_data_combined[x].map(lambda x: np.asarray(x).astype(np.float32))
    
print(train_data_combined.isnull().values.any()  )
  


# test_data = test_data_combined.iloc[:,0]#breaking test data

# test_data_output =test_data_combined.iloc[:,1:]#breaking test data


# train_data_combined = pd.concat([train_data , train_data_combined])
# validation_data_combined = pd.concat([validation_data , validation_data_combined])



datagen = ImageDataGenerator()
train_dataset=datagen.flow_from_dataframe(dataframe=train_data_combined, x_col='filename',y_col = col_list, batch_size = 64, class_mode = 'raw' , resclae = 1/255 )

datagen = ImageDataGenerator()
validation_dataset=datagen.flow_from_dataframe(dataframe=validation_data_combined, x_col='filename',y_col = col_list, batch_size = 64, class_mode = 'raw' , resclae = 1/255 )


# datagen = ImageDataGenerator()
# test_generator=datagen.flow_from_dataframe(dataframe=test_data,batch_size = 64)


#Making the model
CNN = tf.keras.models.Sequential()
CNN.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu',padding='same',input_shape=[None , None , 3]))

CNN.add(tf.keras.layers.BatchNormalization())
CNN.add(tf.keras.layers.LeakyReLU(alpha=0.1))
CNN.add(tf.keras.layers.MaxPool2D(pool_size=2))
CNN.add(tf.keras.layers.Dropout(0.25))

CNN.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu',padding='same'))
CNN.add(tf.keras.layers.BatchNormalization())
CNN.add(tf.keras.layers.LeakyReLU(alpha=0.1))
CNN.add(tf.keras.layers.MaxPool2D(pool_size=2))
CNN.add(tf.keras.layers.Dropout(0.25))


CNN.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu',padding='same'))
CNN.add(tf.keras.layers.BatchNormalization())
CNN.add(tf.keras.layers.LeakyReLU(alpha=0.1))
CNN.add(tf.keras.layers.MaxPool2D(pool_size=2))
CNN.add(tf.keras.layers.GlobalMaxPooling2D())
CNN.add(tf.keras.layers.Dropout(0.25))
CNN.add(tf.keras.layers.Flatten())
#Connecting
CNN.add(tf.keras.layers.Dense(units=1024, activation='relu'))
#Output layer
CNN.add(tf.keras.layers.Dense(units=3, activation='relu'))


from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
filepath=r'C:\Users\Restandsleep\Desktop\VIT\Personal\Handrecogweights.h5'
checkpoint = ModelCheckpoint(filepath,monitor='val_accuracy',mode='max',save_best_only=True,verbose=1)
lrp = ReduceLROnPlateau(monitor='val_loss', factor=0.99, patience=3)
callbacks=[checkpoint,lrp]

CNN.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

print("\n  here \n")


print("\n  here \n")

train_steps=train_dataset.samples//64
validation_steps=validation_dataset.samples//64

history=CNN.fit_generator(
    train_dataset,
    steps_per_epoch=train_steps,
    epochs=25,
    validation_data = validation_dataset,
    validation_steps=validation_steps,
    callbacks=callbacks
)

CNN.save(r'C:\Users\Restandsleep\Desktop\VIT\Personal\Handrecog\model')



   
    
    

    


