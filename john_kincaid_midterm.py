#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: John Kincaid
Deep Learning, Midterm Project
Nov 18, 2018

This network is made of 4 convolutional (3x3) blocks followed by two fully connected hideden layers.
It uses some pre-trained weights from imagenet VGG16, and makes use of data augmentation.

This network unfortunately does not produce good accuracy in my tests. 
I am submitting this network over others that give slightly higher accuracy because I
believe this network "should" work better than them. Those other networks are probably
giving better accuracy because of high recall on the black "background" images.

"""

from keras import applications
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, Model
from keras.layers.core import Dense
from keras.layers import Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD, RMSprop
from keras.utils import to_categorical
import numpy as np
import os
import cv2

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
                if normalized:
                    #pix = cv2.normalize(pix,None, norm_type = cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                    pix = pix/255.0
                all_images.append(pix)
                labels.append(label)
    all_images = np.asarray(all_images)
    return all_images, labels

test_images, test_labels = import_images(TEST_PATH, normalized = True)
encoder = LabelEncoder()
encoder.fit(test_labels)
test_labels = encoder.fit_transform(test_labels)

testY = to_categorical(test_labels, 3)
test_images = test_images.reshape(test_images.shape[0], 256, 256, 3)



#Because we have limited data, do some data augmentation on the data to improve accuracy.
#The nature of the data means some kinds of augmentation, like zoom, are probably not appropriate.
train_datagen = ImageDataGenerator(
		rescale=1./255,
		horizontal_flip=True,
		vertical_flip= True,
		rotation_range= 60,
		#zca_whitening = True,	
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


input_shape = (256, 256, 3)

#I bring in weights from a pretrained network (VGG16)
base_model = applications.vgg16.VGG16(weights= "imagenet", include_top = False, input_shape = input_shape)
#model = applications.resnet50.ResNet50(weights ="imagenet", include_top = False, input_shape = input_shape)

x = base_model.output
#x = Conv2D(512, kernel_size = (3,3), padding = 'same', activation = 'relu')(x)
#x = Conv2D(512, kernel_size = (3,3), padding = 'same', activation = 'relu')(x)
#x = Conv2D(512, kernel_size = (3,3), padding = 'same', activation = 'relu')(x)
#x = MaxPooling2D(pool_size =(2,2))(x)
#x = Dropout(0.25)(x)
x = Flatten()(x)
x = Dense(1024, activation = "relu")(x)
x = Dropout(0.50)(x)
x = Dense(1024, activation = "relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(3, activation="softmax")(x)
model = Model(input=base_model.input, output = predictions)

for layer in base_model.layers[:15]:
	layer.trainable = False
#I create my own network that I will train on. The bottom layers of this network match

#I import the pretrained weights from the top two convolutional blocks of VGG16
#and freeze them.
#The logic for doing this is that the bottom of the nueral network is the most generalized.
#As the convolutional layers progress, the network will learn more sophisticated features.
#Because imagenet is trained on a very different kind of data, I want to train a new network.
#But I think the lowest levels of the imagenet trained network may still be useful.
#for i in range(6):
#	t_weights = base_model.layers[i+1].get_weights()
#	model_t.layers[i].set_weights(t_weights)

#freeze the top two convolutional blocks
#for layer in base_model.layers:
#	layer.trainable = True


#I choose Adam optimizer with a learning rate of 0.01 initally.
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.0001),
              metrics=['accuracy'])
model.summary()
#saves the best model based on validation accuracy.
checkpoint = ModelCheckpoint("jk_checkpoint_1.h5", monitor = 'val_acc', verbose = 1, save_best_only = True, mode= 'auto')

#I run a small number of epochs with a learning rate of 0.01 and the two imported convolution blocks frozen.
model.fit_generator(train_generator,
           steps_per_epoch = 3854//64,
           epochs=15,
           verbose=1,
           validation_data=validation_generator,
           callbacks = [checkpoint])

#for layer in base_model.layers[11:]:
for layer in base_model.layers:
	layer.trainable = True

#set learning rate to 0.001 (down from 0.01)
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr= 0.00001),
              metrics=['accuracy'])
model.summary()

#Train the model for many more epochs with all layers unfrozen.
model.fit_generator(train_generator,
           steps_per_epoch = 3854//32,
           epochs=15,
           verbose=1,
           validation_data=validation_generator,
           callbacks = [checkpoint])

#unfreeze all layers
#for layer in model_t.layers:
#	layer.trainable = True
#for layer in base_model.layers[7:]:
#	layer.trainable = True

#set learning rate to 0.001 (down from 0.01)
#model.compile(loss='categorical_crossentropy',
#		optimizer=Adam(lr=.00001),
#		metrics=['accuracy'])

#Train the model for many more epochs with all layers unfrozen.
#model.fit_generator(train_generator,
#           steps_per_epoch = 3854//32,
#           epochs=10,
#           verbose=1,
#           validation_data=validation_generator,
 #          callbacks = [checkpoint])
#
#for layer in base_model.layers:
#	layer.trainable = True

#set learning rate to 0.001 (down from 0.01)
#model.compile(loss='categorical_crossentropy',
#              optimizer=Adam(lr= 0.000001),
#              metrics=['accuracy'])
#model.summary()

#Train the model for many more epochs with all layers unfrozen.
#model.fit_generator(train_generator,
#           steps_per_epoch = 3854//32,
#           epochs=25,
#           verbose=1,
#           validation_data=validation_generator,
#           callbacks = [checkpoint])

score = model.evaluate(test_images, testY, verbose=0)
#save results
model.save('jk_model_2.h5')
print('Test loss:', score[0])
print('Test accuracy:', score[1])
