# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 21:53:11 2017

@author: Haoyu
"""

import csv
import cv2
import numpy as np
import tensorflow as tf
import keras

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

lines = []

with open('C:\Anaconda3\envs\CarND-Behavioral-Cloning-P3\Rec_Data\driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
images = []
measurements = []

###This part of code only considers center camera images
#for line in lines:
#    source_path = line[0]
#    filename = source_path.split("\\")[-1]
#    current_path = 'C:\\Anaconda3\\envs\\CarND-Behavioral-Cloning-P3\\Rec_Data\\IMG\\' + filename
#    image = cv2.imread(current_path)
#    images.append(image)
#    measurement = float(line[3])
#    measurements.append(measurement)

###This part of code considers 3 cameras
correction = 0.15
for line in lines:
   for i in range(3):
       source_path = line[i]
       filename = source_path.split("\\")[-1]
       current_path = 'C:\\Anaconda3\\envs\\CarND-Behavioral-Cloning-P3\\Rec_Data\\IMG\\' + filename
       image = cv2.cvtColor(cv2.imread(current_path),cv2.COLOR_BGR2RGB)
       if i==0:
           measurement = float(line[3])
       elif i==1:
           measurement = float(line[3]) + correction
       elif i==2:
           measurement = float(line[3]) - correction
       images.append(image)
       measurements.append(measurement)
    
augmented_images, augmented_measurements = [], []

for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

model = Sequential()
model.add(Lambda(lambda x: (x/255.0)-0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
"""
LeNet implemented here, comment line 73 to 81 if using Nvidia NN. Vice Versa
"""
#model.add(Convolution2D(28,5,5,activation='relu',input_shape=(160,320,3) ))
#model.add(MaxPooling2D()) #2x2 stride, valid padding, output size
#model.add(Convolution2D(16,5,5,activation='relu'))
#model.add(MaxPooling2D())
#model.add(Flatten())
#model.add(Dense(200))
##model.add(Dropout(0.5))
#model.add(Dense(50))
#model.add(Dense(1))
"""
NVidia NN implemented here
"""
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
"""
End of model construction
"""
model.compile(optimizer='adam', loss='mse')

model.fit(X_train,y_train, validation_split=0.2, shuffle=True, nb_epoch=3)

model.save('model1.h5')
exit()