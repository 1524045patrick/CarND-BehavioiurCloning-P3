# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains starting files for the Behavioral Cloning Project.

In this project, you will use what you've learned about deep neural networks and convolutional neural networks to clone driving behavior. You will train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle.

We have provided a simulator where you can steer a car around a track for data collection. You'll use image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track.

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Behavioral-Cloning-P3/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

To meet specifications, the project will require submitting five files: 
* model.py (script used to create and train the model)
* drive.py (script to drive the car - feel free to modify this file)
* model.h5 (a trained Keras model)
* a report writeup file (either markdown or pdf)
* video.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)

This README file describes how to output the video in the "Details About Files In This Directory" section.


### Dependencies
This lab requires:

* <b>python3
* <b>keras 2
* <b>numpy
* <b>opencv
* <b>tensorflow

The following resources can be found in this github repository:
* drive.py
* video.py
* writeup_template.md

The simulator can be downloaded from the classroom. In the classroom, we have also provided sample data that you can optionally use to help train your model.

## _Details About Files In This Directory_

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```
## _Code changes for the existing neural network_
##### The drive.py file has been changed in the way that, images captured will be resized automatically to fit the shape for the first layer (lambda normalisation). Everything else stayed the same way. 


Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

#### Why create a video

1. It's been noted the simulator might perform differently based on the hardware. So if your model drives succesfully on your machine it might not on another machine (your reviewer). Saving a video is a solid backup in case this happens.
2. You could slightly alter the code in `drive.py` and/or `video.py` to create a video of what your model sees after the image is processed (may be helpful for debugging).
--
<br>

## _Model Creation_

### Which model was used?
####for this task, I haven chosen the Architecture from Nvidia for the exact same task. After testing it, I noticed that it was very prone to overfitting the data. So additional Dropout-Layers in the Conv2D-Group made the job. For further informations on the model Architecture check the model on tensorboard.
###`tensorboard --logdir=fullpath/logs/`
####This model drives stable at 20mph. 
--

## _Data Preprocessing and Characteristics_

#### The main problem of this project was the data. Since the recording of my own data did not result well, for this project I used the data from udacity.

####1. Here is a problem, the main part of the data is with car just driving straight. Leading into biases near 0 or even 0.
####2. Not enough training data, the dataset is really small so we need to create additional data
--

## _Solution to these problems_

####1. Udacity - Dataset, the udacity dataset has almost no values with a steering angle of 0, in addition to the adjustment angle. Hence the biases won't get to close to 0 .
####2. I created 2 functions to create augmented data, these modified the existing dataset and thus created additional data

--
### `random_brightness(image)`

*  This function takes an image as input, the image is then converted to HSV. Since it is easy to randomely modify the brightness in HSV colorspace, I here randomely change the brightness of the last (Value) channel. After having randomely modified the brightness of the image is being converted back to RGB colorspace and returned as return value.

--

### `flip(image, angle)`
* for additional data augmentation inside the pipeline, randomely images are vertically flipped and the corresponding label to the images is multiplied by `-1`. This returns later the _flipped image_ and the new _angle_.


# CarND-BehavioiurCloning-P3
