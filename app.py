import matplotlib.pyplot as plt
import tensorflow as tf
from imutils import paths
from tensorflow import keras

images = sorted(list(paths.list_files('Pnuemonia/')))

import cv2

print(images)

test_images = cv2.imread(images[1])
print(type(test_images))
print(test_images.shape)

# Min and Max Resolutions
width = []
height = []
for image in images:
    if image.endswith('.jpg'):
        targetimage = cv2.imread(image)
        width.append(targetimage.shape[0])
        height.append(targetimage.shape[1])

print(len(width))
print(len(height))

print(min(width), min(height))
print(max(width), max(height))

import random
features = []
labels = []

random.shuffle(images)

print(images)

for image in images:
    if image.endswith('.jpg'):
        targetimage = cv2.imread(image)
        # targetimage=cv2.cvtColor(targetimage,cv2.COLOR_BAYER_BG2RGB)
        targetimage = cv2.resize(targetimage, (256, 256)).flatten()
        features.append(targetimage)
        labels.append(image.split('/')[1])

print(features)

print(labels)

import numpy as np
features = np.array(features)
labels = np.array(labels)

print(features)

features = features / 255.0

print(features)

from sklearn.model_selection import train_test_split

(trainX, testX, trainY, testY) = train_test_split(
    features, labels, test_size=0.20)

print(trainX.shape)
print(trainX.shape)
print(trainY)

# One Hot Encoding
from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

print(trainY)

# CNN
from keras.preprocessing.image import ImageDataGenerator

training_generator = ImageDataGenerator(
    rescale=1 / 255.0, validation_split=0.25)

type(training_generator)

training = training_generator.flow_from_directory(
    'Pnuemonia/', target_size=(256, 256), batch_size=32, class_mode='categorical', subset='training')

validation = training_generator.flow_from_directory('Pnuemonia/', target_size=(
    256, 256), batch_size=32, class_mode='categorical', subset='validation')

#import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.optimizers.schedules import ExponentialDecay

model = Sequential()
# First COnv Layer
model.add(Conv2D(16, (3, 3), input_shape=(256, 256, 3), activation='relu'))
model.add(MaxPool2D(2, 2))
model.add(Dropout(0.20))
# Second Conv Layer
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPool2D(2, 2))
model.add(Dropout(0.20))

# third Conv Layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D(2, 2))
model.add(Dropout(0.20))

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.summary()

adam = tf.keras.optimizers.Adam(learning_rate=0.001)

lr_schedule = ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.9)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

earlystopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=5, verbose=2,
    mode='auto', baseline=None, restore_best_weights=True
)

model.compile(
    optimizer=adam,
    loss='categorical_crossentropy',
    metrics=['accuracy'])

callbacks_list = [earlystopping]
history = model.fit(training, validation_data=validation,
                    epochs=20, callbacks=[callbacks_list])

print(targetimage)
print(validation.labels)

# tests
targetimage = cv2.imread('LungXray.jpg')
targetimage = cv2.resize(targetimage, (256, 256))

print(targetimage.shape)

model.predict(targetimage.reshape(-1, 256, 256, 3))

model.predict(testX[10].reshape(-1, 256, 256, 3))

print(testY[10])

model.predict(testX[78].reshape(-1, 256, 256, 3))

print(testY[78])
