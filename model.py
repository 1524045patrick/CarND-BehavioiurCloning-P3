import matplotlib.image as mpimg
import numpy as np
import cv2
import pandas
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import math

flags = tf.app.flags
FLAGS = flags.FLAGS

# DEFINE FLAGS VARIABLES#
flags.DEFINE_float('steering_adjustment', 0.27, "Adjustment angle.")
flags.DEFINE_integer('epochs', 9, "The number of epochs.")
flags.DEFINE_integer('batch_size', 30, "The batch size.")  # *10
# PART 1: Data Preprocessing

# importing columns from driving_log
colnames = ['center', 'left', 'right',
            'steering', 'throttle', 'brake', 'speed']
data = pandas.read_csv('data/driving_log.csv', skiprows=[0],
                       names=colnames)

center = data.center.tolist()
center_recover = data.center.tolist()
left = data.left.tolist()
right = data.right.tolist()
steering = data.steering.tolist()
steering_recover = data.steering.tolist()

#  Shuffle center and steering. Use 10% of central images and steering angles for validation.
center, steering = shuffle(center, steering)
center, X_valid, steering, y_valid = train_test_split(
    center, steering, test_size=0.10)

# Filtering straight, left and right images
d_straight, drive_left, drive_right = [], [], []
angle_straight, angle_left, angle_right = [], [], []
for i in steering:
    # Positive angle is turning from Left to Right. Negative is turning from Right to Left#
    index = steering.index(i)
    if i > 0.15:
        drive_right.append(center[index])
        angle_right.append(i)
    if i < -0.15:
        drive_left.append(center[index])
        angle_left.append(i)
    else:
        d_straight.append(center[index])
        angle_straight.append(i)

# Added Recovery for recovering when leaving the road
ds_size, dl_size, dr_size = len(d_straight), len(drive_left), len(drive_right)
main_size = math.ceil(len(center_recover))
l_xtra = ds_size - dl_size
r_xtra = ds_size - dr_size
indice_L = random.sample(range(int(main_size)), l_xtra)
indice_R = random.sample(range(int(main_size)), r_xtra)

"""
Filtering angles with lower values than -0.15 and add into drive left list with the difference from the
adjustment angle. Other values which are bigger than 0.15 will be added into the drive right list with the sum
from the adjustment angle and the steering angle
"""
for i in indice_L:
    if steering_recover[i] < -0.15:
        drive_left.append(right[i])
        angle_left.append(steering_recover[i] - FLAGS.steering_adjustment)
for i in indice_R:
    if steering_recover[i] > 0.15:
        drive_right.append(left[i])
        angle_right.append(steering_recover[i] + FLAGS.steering_adjustment)

# Combination of the images into the training data for having more training data
X_train = d_straight + drive_left + drive_right
y_train = np.float32(angle_straight + angle_left + angle_right)

### Data augmentation ###

"""
This function is used to create random brightness in the image passed in the parameters
image = Image which shall later be passed to the Neural Network
Inside we use the conversion from RGB2HSV => Hue,Saturation,Value
After having randomely changed brightness we simply return the new image
"""


def random_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    rand = random.uniform(0.3, 1.0)
    hsv[:, :, 2] = rand * hsv[:, :, 2]
    preprocessed_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return preprocessed_img


"""
Simple function, takes in an image and the corresponding steering angle to the image
This flips the image and converts the steering angle with the multiplication *(-1)
"""


def flip(image, angle):
    new_image = cv2.flip(image, 1)
    new_angle = angle * (-1)
    return new_image, new_angle


"""
Not wanting to resize (preprocess) the data inside the neural network this function was created
it only keeps the wanted mask of the image (without parts of the car) and then uses the resize
function of opencv to get needed shape for the neural network (64,64)
"""


def crop_resize(image):
    cropped = cv2.resize(image[60:140, :], (64, 64))
    return cropped


"""
Training generator:
Takes in batch_size as parameter, then shuffles the whole dataset (to prevent overfitting).
Each Datapoint in the batch_size will be resized to (64,64)
Angle will be slightly randomely modified (preventing overfitting)
flip_coin is used to randomely flip incoming images in the batch, some will be flipped others won't
Apply random brightness, resize, crop into the chosen sample. Add some small random noise for
chosen angle.
"""


def generator_data(batch_size):
    batch_train = np.zeros((batch_size, 64, 64, 3), dtype=np.float32)
    batch_angle = np.zeros((batch_size,), dtype=np.float32)
    while True:
        data, angle = shuffle(X_train, y_train)
        for i in range(batch_size):
            choice = int(np.random.choice(len(data), 1))
            batch_train[i] = crop_resize(random_brightness(
                mpimg.imread(data[choice].strip())))
            batch_angle[i] = angle[choice] * \
                             (1 + np.random.uniform(-0.10, 0.10))
            flip_coin = random.randint(0, 1)
            if flip_coin == 1:
                batch_train[i], batch_angle[i] = flip(
                    batch_train[i], batch_angle[i])

        yield batch_train, batch_angle


# Validation generator: pick random samples. Apply resizing and cropping on chosen samples
"""
Validation samples will be picked randomely like in generator_data(batch_size)
This function takes in the image data, the corresponding steering angle and the batch_size as parameter
data => String Array with Path to the images
angle => Float Array with Steering Angles
"""


def generator_valid(data, angle, batch_size):
    batch_train = np.zeros((batch_size, 64, 64, 3), dtype=np.float32)
    batch_angle = np.zeros((batch_size,), dtype=np.float32)
    while True:
        data, angle = shuffle(data, angle)
        for i in range(batch_size):
            rand = int(np.random.choice(len(data), 1))
            batch_train[i] = crop_resize(mpimg.imread(data[rand].strip()))
            batch_angle[i] = angle[rand]
        yield batch_train, batch_angle


"""
The Model-Architecture for predicting the steering angle,
inspired by the Model from NVIDIA to predict steering angles with slight modifications
Dropout-Layers where moved into the Conv2D calls, to prevent overfitting on the images
"""


def ModelNvidia():
    from os import path
    from keras.models import load_model
    if path.isfile("./model.h5"):
        print("Loading previous Model with weights")
        model = load_model("./model.h5")
    else:
        input_shape = (64, 64, 3)
        model = Sequential()
        model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=input_shape))
        model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample=(2, 2)))
        model.add(Activation('relu'))
        model.add(Dropout(0.4))
        model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample=(2, 2)))
        model.add(Activation('relu'))
        model.add(Dropout(0.4))
        model.add(Convolution2D(48, 5, 5, border_mode='valid', subsample=(2, 2)))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(2, 2)))
        model.add(Dropout(0.4))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(2, 2)))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(80))
        model.add(Dropout(0.5))
        model.add(Dense(40))
        model.add(Dense(16))
        model.add(Dense(10))
        model.add(Dense(1))
        adam = Adam(lr=0.0001)
        model.compile(optimizer=adam, loss='mse')
        model.summary()
    return model


def testModel():
    from os import path
    from keras import regularizers
    from keras.models import load_model
    from keras.activations import relu
    from keras.layers import Conv2D
    if path.isfile("./model.h5"):
        print("Loading previous Model with weights")
        model = load_model("./model.h5")
    else:
        input_shape = (64, 64, 3)
        model = Sequential()
        model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=input_shape))
        model.add(
            Conv2D(24, (5, 5), activation=relu, padding='valid', strides=(2, 2), kernel_regularizer=regularizers.l2))
        model.add(Dropout(0.4))
        model.add(
            Conv2D(36, (5, 5), activation=relu, padding='valid', strides=(2, 2), kernel_regularizer=regularizers.l2))
        model.add(Dropout(0.4))
        model.add(
            Conv2D(48, (3, 3), activation=relu, padding='valid', strides=(2, 2), kernel_regularizer=regularizers.l2)
        model.add(Dense(60))
        model.add(Dropout(rate=0.4))
        model.add(Dense(10))
        model.add(Dropout(0.4))
        model.add(Dense(1))
        adam = Adam(lr=0.0001))
        model.compile(optimizer=adam, loss='mse')
        model.summary()
    return model


# PART 3: TRAINING
"""Parameter is not used in this setup, but could be for passing arguments from the cmd"""


def main(_):
    # log_dir customized for use in floydhub
    tensorboard = TensorBoard(
        log_dir='/output/logs', histogram_freq=2, write_graph=True, write_images=False)
    data_generator = generator_data(FLAGS.batch_size)
    valid_generator = generator_valid(X_valid, y_valid, FLAGS.batch_size)
    model = testModel()
    model.fit_generator(data_generator, steps_per_epoch=(math.ceil(len(X_train) / FLAGS.batch_size)),
                        nb_epoch=FLAGS.epochs,
                        validation_data=valid_generator, validation_steps=(
            len(X_valid) / FLAGS.batch_size),
                        callbacks=[tensorboard])
    print('Done Training')

    model_json = model.to_json()
    with open("/output/model.json", "w") as json_file:
        json_file.write(model_json)
    model.save("/output/model.h5")
    print("Saved model to disk")


if __name__ == '__main__':
    tf.app.run()
