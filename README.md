# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains important files for the Behavioral Cloning Project:
* model.py (script used to create and train the model)
* drive.py (script to drive the car - feel free to modify this file)
* model.h5 (a trained Keras model)
* writeup_report.md (a report writeup file (either markdown or pdf)
* run.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)

[video]: ./run.mp4 "simulation video"

![alt text][video]

In this project, the training dataset was collected from a Unity simulation driving game. The driving environment is only one lane on the road. There are two lane road which can be trained in further works. I used deep neural networks and convolutional neural networks to clone driving behavior. I also trained, validated and tested a model using Keras. In the testing simulation, the model can output a steering angle to an autonomous vehicle.

Udacity has provided a simulator where I can steer a car around a track for data collection. I used image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track. It would be good to play with joystick so that the driving behaviour will be more appropriate with smoother steering angles and the speed change.

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./report_images/model_architecture.jpeg "Model Architecture Reference"
[image2]: ./report_images/center_drive.jpg "Center Lane Drive"
[image3]: ./report_images/left_recovery1.jpg "Recovery Image"
[image4]: ./report_images/left_recovery2.jpg "Recovery Image"
[image5]: ./report_images//left_recovery3.jpg "Recovery Image"
[image6]: ./report_images/loss_result.png  "Loss Result"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. All required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

The autonomous driving video can be recorded by 
```sh
#record the autonomous mode in 'run1' folder
python drive.py model.h5 run1
#generate the recording video as 'run1.mp4'
python video.py run1
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. 
The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
[1]: https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf "Nvidia reference link"

My model is based on the architecture by NVIDIA's paper [End-to-End Learning for Self-Driving Cars][1]. This is a 
model example:
![alt text][image1] 

My model consists 10 layers including normalization, cropping, convolutions, and fully connected layers. I first 
normalized the data using a Keras lambda layer （model.py lines 96) to set the values between -0.5 and 0.5, which can 
also speed up the GPU processing. The second layer is the cropping （model.py lines 98), thus only the road part is 
remained, and the input information is more targeted. Then I set 3 convolution layers （model.py lines 101-103) with 5x5 
filters and 2x2 strides, followed by 2 convolution layers （model.py lines 104-105) with 3x3 filters and 
non-strides. These are utilized to extract the features. The depth of the convolution layers increases from 3 to 64, 
and the ELU activation instead of RELU is used to make the result more robust. As demonstrated in the NVIDIA's paper,
 the layers number is chosen empirically and ends up with a good simulation result. After the convolution layers, there 
are 3 fully connected layers （model.py lines 109-112) which seems acted like a controller. 


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 106). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 
90). The training and validation images were generated by random batches with size 126. The images were flipped to 
get more data and reduce the bias of the steering angles distribution. 

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 120).

#### 4. Creation of the Training Set & Training Process

Training data was chosen to keep the vehicle driving on the road. When collecting the data, I mainly kept the car in 
the center lane. Three images from left, center, and right cameras were taken to augment the training data as 
well as the driving situation. The steering angles corresponding to images taken by the left and right cameras are 
modified a bit, so that the model can learn to steer the car back to the center. 

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example 
image of center lane driving:

![alt text][image2]

I also recorded the vehicle recovering from the left side and right sides of the road back to center so that the 
vehicle would learn to recover from the side to the lane center and prevent driving off the road. These images show what a 
recovery looks like starting from the right side :

![alt text][image3]
![alt text][image4]
![alt text][image5]

After the collection process, I then preprocessed this data by randomly modifying the
 images brightness. In addition, I flipped all the images and angles, added them to the dataset, which can augment 
 the data. Thus, I had 43590 number of data points


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. 
The generator is used to only get a batch of data during each training iteration, to save the memory. The batch size 
input into the model is actually 126, while the input into the generator is 22, because each sample in csv file 
contains 3 images, and the flipping process in the generator function will twice the size of the data. The ideal number of epochs was 8 as it shows a good result 
and also save training time. I used an adam optimizer so that manually training the learning rate was not necessary. 

The training and validation loss for each epoch is shown as below:
![alt text][image6]


### Autonomous Mode Simulation 

The result of training model can be tested and visualized by an autonomously drive by: 
```sh
python drive.py model.h5
```
If the speed is set to be 10 mph, the car can keep on the lane center and drive smoothly. This simulation video is 
shown in 'run.mp4'. When I set the speed to be 20 mph, the car will still keep inside the lane, but sometimes will 
steer too much so that it will drive left and right when go advance. I think this issue may be due to the lack of 
training data in high speed and with smooth steering angles. I do not have a joystick to control the car, and by the 
keyboard it is hard to control the steering angle in high speed as well as keep it in the lane. Therefore, the data is collected 
mainly under a speed of 12mph. 

For the further improvement, the training on track two can be implemented and the model can be tested and modified. A
 joystick can be bought to control the car, which can reach a better result when driving in higher speed.
