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
from random import shuffle
import sklearn

lines = []

with open('C:\Anaconda3\envs\CarND-Behavioral-Cloning-P3\Rec_Data\driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

correction = 0.15
def generator(samples, batch_size=32):
    correction = 0.15
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0,num_samples,batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            augmented_images, augmented_measurements = [], []
            for batch_sample in batch_samples:
                for i in range(3):
                    source_path = batch_sample[i]
                    filename = source_path.split("\\")[-1]
                    current_path = 'C:\\Anaconda3\\envs\\CarND-Behavioral-Cloning-P3\\Rec_Data\\IMG\\' + filename
                    image = cv2.cvtColor(cv2.imread(current_path),cv2.COLOR_BGR2RGB)
                    if i==0:
                        measurement = float(batch_sample[3])
                    elif i==1:
                        measurement = float(batch_sample[3]) + correction
                    elif i==2:
                        measurement = float(batch_sample[3]) - correction
                    images.append(image)
                    measurements.append(measurement)
            for image, measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image,1))
                augmented_measurements.append(measurement*-1.0)
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train,y_train)
            
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

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

model.fit_generator(train_generator, samples_per_epoch=7200,validation_data=validation_generator,nb_val_samples=1792, nb_epoch=3)

model.save('model.h5')
exit()