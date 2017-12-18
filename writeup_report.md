# **Behavioral Cloning** 

## Writeup Report

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/Steering_Histogram.JPG "Steering Histogram"
[image2]: ./examples/3_cams.png "3 cameras"
[image3]: ./examples/center_driving.jpg "center driving"
[image4]: ./examples/recov1.jpg "Recovery Image 1"
[image5]: ./examples/recov2.jpg "Recovery Image2"
[image6]: ./examples/recov3.jpg "recov Image3"
[image7]: ./examples/curve.jpg "Curved road Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed these points in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* video.mp4 showing how well the trained model drives the car autonomously
* writeup_report.md (which is this file) summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training, validating and saving the model.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model adopted the popular neural network architecture developed by Nvidia. It consists of 5 convolutional layers and 3 fully connected layers. (model.py lines 86-95) 

The model includes RELU layers to introduce nonlinearity (code line 86-90), and the data is normalized in the model using a Keras lambda layer (code line 69). 

#### 2. Attempts to reduce overfitting in the model

I didn't put dropout layers in my architecture as people commonly do to reduce overfitting, due to the fact that dropout layers will weaken the influence of training targets that are far from 0. From the following statistical histogram of steering angle values in the training data, it's clear that most of training data fall around 0 steering. However, steering angles that are far from 0 are useful when training the model, especially for curved sections of the road.

![alt text][image1]

Therefore, first of all, I collected enough data and augmented the data to reduce overfitting (code line 56-60).

Secondly, I trained and validated the model on different data sets to ensure that the model was not overfitting (code line 29,65-66). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 99).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and emphasizing on collected curved road data to obtain appropriate raw data.

Then I used a series of techniques, including shuffling, cropping, flipping and augmentation, to process the data.

For details about how I processed the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to chose an easy start and improve step by step.

My first step was to use LeNet which was implemented in the previous project. I thought this model might be appropriate because it was intended for recognizing traffic signs. However, it didn't work well, partly because the architecture was intented to classify different classes, not to give continuous steering output, partly because the training data was not sufficient (I simply drove in the center of the road to collect data). The collected images look like this:

![alt text][image3]

The car goes out of the track immediately.

To get a better performance, I changed the model to Nvidia's architecture provided in the instructions. Along with this, I collected more data, especially for curved sections. The curve road data looked like:

![alt text][image7]

I also augmented the data by flipping the images and corresponding steering angles:
```python
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)
```

Then I trained the model and tested in the simulator, still not good. The car goes out of the track on sharp curves due to insufficient steering inputs. I collected more data on the curves for training, but still didn't help. Then I realized I must use some other techniques.

So first I collected lots of recovery driving data:

![alt text][image4]

![alt text][image5]

![alt text][image6]

I also coverted the training images to the same color space as the test environment using:
```python
image = cv2.cvtColor(cv2.imread(current_path),cv2.COLOR_BGR2RGB)
```
Then I used images from left and right cameras to help training. This method was also recommened in the instructions.

![alt text][image2]

A generator was used to speed up training. I finally randomly shuffled the data set and put Y% of the data into a validation set. So the training data generation now looks like this:
```python
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
```

At the end of the process, the vehicle is able to drive autonomously around the track while staying in the center of the road. The results is shown in [video.mp4](https://www.youtube.com/watch?v=kEU_pF6PxdY)

#### 2. Final Model Architecture

The final model architecture (model.py lines 68-94) is shown in the following chart:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   							| 
| Layer 1: Normalization     	| Lambda normalization, outputs 160x320x3 	|
| Layer 2: Cropping				  	|	 cuts off top and bottom portion of images	|
| Layer 3: Convolution 5x5	    | 2x2 stride, valid padding, output depth 24, RELU 	|
| Layer 4: Convolution 5x5	    | 2x2 stride, valid padding, output depth 36, RELU  |
| Layer 5: Convolution 5x5	    | 2x2 stride, valid padding, output depth 48, RELU  |
| Layer 6: Convolution 3x3	    | 1x1 stride, valid padding, output depth 64, RELU  |
| Layer 7: Convolution 3x3	    | 1x1 stride, valid padding, output depth 64, RELU  |
| Layer 8: Flatten	      	| prepare for fully connected layers 	|
| Layer 9: Fully Connected   | output size 100		|
| Layer 10: Fully Connected   | output size 50		|
| Layer 11: Fully Connected   | output size 10		|
| Layer 12: Fully Connected   | final output size 1 |

