#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 18:16:16 2018

@author: sharonkincaid
"""

from keras import applications
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from sklearn.metrics import classification_report
from keras.models import Sequential, Model
from keras.layers.core import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, RMSprop, Adagrad
import numpy as np

import os
from PIL import Image
import cv2
from sklearn.preprocessing import normalize

from sklearn.preprocessing import LabelEncoder

TRAIN_PATH = 'midterm_project/train'
TEST_PATH = 'midterm_project/test'
def import_images(local_path, normalized = False):
    all_images = list()
    labels = []
    for root, subdir_list, filenames in os.walk(local_path):
        label = os.path.basename(root)
        for file in filenames:
            if '.png' in file:
                pix = cv2.imread(os.path.join(root,file))
		#in_image = Image.open(os.path.join(root, file))
                #small_image = in_image.resize((32,32))
                #small_image = small_image.convert('RGB')
                #pix = in_image.getdata()
                if normalized:
                    #pix = cv2.normalize(pix,None, norm_type = cv2.NORM_MINMAX,                            dtype=cv2.CV_32F)
                    pix = pix/255.0
                all_images.append(pix)
                labels.append(label)
    all_images = np.asarray(all_images)
    return all_images, labels

#train_images, train_labels = import_images(TRAIN_PATH, normalized = True)
#label_dict = dict(zip(encoder.transform(encoder.classes_), encoder.classes_))
#train_labels = encoder.fit_transform(train_labels)


test_images, test_labels = import_images(TEST_PATH, normalized = True)
encoder = LabelEncoder()
encoder.fit(test_labels)
test_labels = encoder.fit_transform(test_labels)
#images = images.reshape(images.shape[0], 32, 32, 3)

#from sklearn.model_selection import train_test_split
#(trainX, testX, trainY, testY) = train_test_split(images, labels,
#	test_size=0.25, random_state=42)


test_images = test_images.reshape(test_images.shape[0], 256, 256, 3)
#train_images = train_images.reshape(train_images.shape[0], 256, 256, 3)





from keras.utils import to_categorical

input_shape = (256, 256, 3)

base_model = applications.vgg16.VGG16(weights= "imagenet", include_top = False, input_shape = input_shape)

#for layer in base_model.layers:
#	layer.trainable = False

#for x in range(12):
#	base_model.layers.pop()
#base_model.summary()
#trainY = to_categorical(train_labels, 3)
testY = to_categorical(test_labels, 3)
from keras.layers import Dropout, Flatten
from keras.layers import GlobalAveragePooling2D, Conv2D, MaxPooling2D

#model = Sequential()
#model.add(Conv2D(128, kernel_size=(3, 3),
#                 padding='valid',
#                 activation='relu',
#                 ))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
#model.add(Flatten())
#model.add(Dense(1024, activation='relu'))
#model.add(Dropout(0.25))
#model.add(Dense(1025, activation='relu'))

#base_model.summary()
x = base_model.output
#x = GlobalAveragePooling2D()(x)
#x = Conv2D(256, kernel_size=(3, 3),
#                 padding='same',
#                 activation='relu',
#                 )(x)
#x = Conv2D(256, kernel_size=(3, 3),
#                 padding='same',
#                 activation='relu',
#                 )(x)
#x = Conv2D(256, kernel_size=(3, 3),
#                 padding='same',
#                 activation='relu',
#                 )(x)
#model.add(MaxPooling2D(pool_size=(2, 2)))
#x = MaxPooling2D(pool_size=(2,2))(x)
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
#x = Dense(1024, activation = 'relu')(x)
predictions = Dense(3, activation='softmax')(x)

model = Model(input = base_model.input, output = predictions)
print('test')

model_t = Sequential()
#block 1, pretrained weights
model_t.add(Conv2D(64, kernel_size = (3,3), padding='same', activation='relu',
		input_shape=(256,256,3)))
model_t.add(Conv2D(64, kernel_size = (3,3), padding='same', activation='relu',
		))
model_t.add(MaxPooling2D(pool_size=(2,2)))
#block 2, pretrained weights
model_t.add(Conv2D(128, kernel_size = (3,3), padding='same', activation='relu',
		))
model_t.add(Conv2D(128, kernel_size = (3,3), padding='same', activation='relu',
		))
model_t.add(MaxPooling2D(pool_size=(2,2)))
#block 3, random weights
model_t.add(Conv2D(256, kernel_size = (3,3), padding='same', activation='relu',
		))
model_t.add(Conv2D(256, kernel_size = (3,3), padding='same', activation='relu',
		))
model_t.add(Conv2D(256, kernel_size = (3,3), padding='same', activation='relu',
		))
model_t.add(MaxPooling2D(pool_size=(2,2)))
model_t.add(Dropout(.25))
#block 4, random weights
model_t.add(Conv2D(256, kernel_size = (3,3), padding='same', activation='relu',
		))
model_t.add(Conv2D(256, kernel_size = (3,3), padding='same', activation='relu',
		))
model_t.add(Conv2D(256, kernel_size = (3,3), padding='same', activation='relu',
		))
model_t.add(MaxPooling2D(pool_size=(2,2)))
model_t.add(Dropout(.25))
model_t.add(Flatten())
model_t.add(Dense(256, activation = 'relu'))
model_t.add(Dropout(.25))
model_t.add(Dense(256, activation = 'relu'))
model_t.add(Dropout(.5))
model_t.add(Dense(3, activation='softmax'))
for i in range(6):
	t_weights = base_model.layers[i+1].get_weights()
	#print(len(test), base_model.layers[i+1].name, model_t.layers[i].name)
	model_t.layers[i].set_weights(t_weights)
	#model_t.layers[i] = base_model.layers[i+1]

for layer in model_t.layers[:6]:
	layer.trainable = False

for lay in model.layers[12:]:
	layer.trainable = False	

#model_t.layers[0].set_weights(base_model.layers[1].get_weights)

#model_t.layers[1].set_weights(base_model.layers[2].get_weights)

#model_t.layers[3].set_weights(base_model.layers[4].get_weights)
#model_t.layers[4].set_weights(base_model.layers[5].get_weights)
#model_t.layers[0].set_weights(test)
#model_t.summary()
model_t.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.01),
              metrics=['accuracy'])

from keras.applications.vgg16 import preprocess_input

train_datagen = ImageDataGenerator(
		rescale=1./255,
		#preprocessing_function = preprocess_input(),
		horizontal_flip=True,
		vertical_flip= True,
		rotation_range=30,
		)
test_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 32
train_generator = train_datagen.flow_from_directory(
	TRAIN_PATH,
	target_size =(256,256),
	batch_size = batch_size,
	class_mode = 'categorical')
validation_generator = test_datagen.flow_from_directory(
	TEST_PATH,
	target_size =(256,256),
	batch_size = batch_size,
	class_mode = 'categorical')

checkpoint = ModelCheckpoint("jk_checkpoint_1.h5", monitor = 'val_acc', verbose = 1, save_best_only = True, mode= 'auto')


model_t.fit_generator(train_generator,
           steps_per_epoch = 3854//64,
           epochs=20,
           verbose=1,
           validation_data=validation_generator,
           callbacks = [checkpoint])

#unfreeze all layers
for layer in model_t.layers:
	layer.trainable = True

model_t.summary()

model_t.compile(loss='categorical_crossentropy',
              optimizer=Adam(0.001),
              metrics=['accuracy'])

model_t.fit_generator(train_generator,
           steps_per_epoch = 3854//64,
           epochs=100,
           verbose=1,
           validation_data=validation_generator,
           callbacks = [checkpoint])

#for i, layer in enumerate(base_model.layers):
#	if layer.trainable == True:
#		print(i,layer.name)
score = model_t.evaluate(test_images, testY, verbose=0)
model_t.save('jk_model.h5')
print('Test loss:', score[0])
print('Test accuracy:', score[1])
