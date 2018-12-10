from scipy import ndimage
import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers import Flatten, Dense, Lambda
from keras.layers.core import Dropout


def random_brightness(image):
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_bright = 0.8 + 0.4 * (2 * np.random.uniform() - 1.0)
    image1[:, :, 2] = image1[:, :, 2] * random_bright
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return image1


def preprocess(images, measurements):
    # flip the images and measurements
    steering_angles = []
    images_pro = []
    for measurement in measurements:
        steering_angles.append(measurement)
        measurement_flipped = -measurement
        steering_angles.append(measurement_flipped)

    for image in images:
        # change the brightness
        image1 = random_brightness(image)
        images_pro.append(image1)

        # flip the image
        image_flipped = np.fliplr(image)
        images_pro.append(image_flipped)

    return images_pro, steering_angles


def generator(samples, sample_size=21):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            measurements = []

            for line in batch_samples:
                for i in range(3):
                    source_path = line[i]
                    filename = source_path.split('/')[-1]
                    #I have moved the data folder under the opt folder
                    # current_path = '../../opt/data/IMG/'+filename
                    current_path = './data/IMG/' + filename
                    image = ndimage.imread(current_path)
                    images.append(image)

                    # create adjusted steering measurements for the side camera images
                    correction = 0.2  # this is a parameter to tune
                    if i == 0:
                        angle = float(
                            line[3])  # steering angle for centre camera image
                    elif i == 1:
                        angle = float(line[
                                          3]) + correction  # steering angle for left camera image
                    else:
                        angle = float(line[
                                          3]) - correction  # steering angle for right camera image

                    measurements.append(angle)

            # this will twice the samples size
            X_train, y_train = preprocess(images, measurements)
            X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
            yield (np.array(X_train), np.array(y_train))


lines = []
#I have moved the data folder under the opt folder
# with open('../../opt/data/driving-log.csv', 'r') as f:
with open('./data/driving_log.csv', 'r') as f:
    reader = csv.reader(f)
    for line in reader:
        lines.append(line)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)
print(len(train_samples))
print(len(validation_samples))

model = Sequential()
# normalize the images
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3), name = 'Normalization'))
# crop the iamge
model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3), name = 'Cropping'))
# model architecture
# convolution
model.add(Conv2D(24, 5, strides=(2, 2), activation="elu", name ='Conv1')) #output = 158x43x24
model.add(Conv2D(36, 5, strides=(2, 2), activation="elu", name = 'Conv2' )) #output = 77x22x36
model.add(Conv2D(48, 5, strides=(2, 2), activation="elu", name = 'Conv3')) #output = 37x7x64
model.add(Conv2D(64, 3, activation="elu", name = 'Conv4' )) #output = 35x7x64
model.add(Conv2D(64, 3, activation="elu", name = 'Conv5')) #output = 32x5x64
model.add(Flatten(name ='Flat1')) #output = 10240
model.add(Dropout(0.5, name = 'Dropout1'))
# fully connection
model.add(Dense(100, name = 'FullyCon1')) #output = 100
model.add(Dropout(0.5, name = 'Dropout2'))
model.add(Dense(50, name = 'FullyCon2')) #output = 50
model.add(Dense(10, name = 'FullyCon3')) #output = 10
model.add(Dense(1, name = 'Output')) #output = 1
plot_model(model,to_file='model.png') #visualize the model

# generate the training and validation dataset, the generator output has the
# twice size as the batch_size
train_generator = generator(train_samples, sample_size=21)
validation_generator = generator(validation_samples, sample_size=21)

# compile and fit the model
model.compile('adam', 'mse')
batch_size = 126
history_object = model.fit_generator(train_generator, steps_per_epoch=
int(2*len(train_samples) / batch_size), validation_data=validation_generator,
                                     validation_steps=int(2*len(validation_samples) / batch_size),
                                     epochs=8, verbose=1)

model.save('./model1.h5')

# visualize the training and validation loss
### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
