#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 17:55:48 2018

@author: sharonkincaid
"""

import numpy as np
import time
from tensorflow.examples.tutorials.mnist import input_data

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, RMSprop
from keras import optimizers
from keras import backend as K

import matplotlib.pyplot as plt
import os
import cv2

def default_discriminator(input_shape = (64,64,1)):

    Discriminator = Sequential()
    depth = 64
    dropout = 0.5
            # In: 28 x 28 x 1, depth = 1
            # Out: 14 x 14 x 1, depth=64
    Discriminator.add(Conv2D(depth*1, 7, strides=2, input_shape=input_shape,\
                padding='same'))
    Discriminator.add(LeakyReLU(alpha=0.2))
    Discriminator.add(Dropout(dropout))

    Discriminator.add(Conv2D(depth*2, 3, strides=1, padding='same'))
    Discriminator.add(LeakyReLU(alpha=0.2))
    Discriminator.add(Dropout(dropout))

    Discriminator.add(Conv2D(depth*2, 5, strides=2, padding='same'))
    Discriminator.add(LeakyReLU(alpha=0.2))
    Discriminator.add(Dropout(dropout))


    Discriminator.add(Conv2D(depth*4, 3, strides=1, padding='same'))
    Discriminator.add(LeakyReLU(alpha=0.2))
    Discriminator.add(Dropout(dropout))

    Discriminator.add(Conv2D(depth*4, 5, strides=2, padding='same'))
    Discriminator.add(LeakyReLU(alpha=0.2))
    Discriminator.add(Dropout(dropout))


    Discriminator.add(Conv2D(depth*8, 3, strides=1, padding='same'))
    Discriminator.add(LeakyReLU(alpha=0.2))
    Discriminator.add(Dropout(dropout))

#    Discriminator.add(Conv2D(depth*8, 3, strides=1, padding='same'))
#    Discriminator.add(LeakyReLU(alpha=0.2))
#    Discriminator.add(Dropout(dropout))

#    Discriminator.add(Conv2D(depth*8, 3, strides=1, padding='same'))
#    Discriminator.add(LeakyReLU(alpha=0.2))
#    Discriminator.add(Dropout(dropout))

    Discriminator.add(Flatten())
    Discriminator.add(Dense(1))
    Discriminator.add(Activation('sigmoid'))
    return Discriminator

####

def default_generator():
    Generator = Sequential()
    dropout = 0.5
    depth = 256
    dim = 16

    Generator.add(Dense(dim*dim*depth, input_dim=100))
    Generator.add(BatchNormalization(momentum=0.9))
    Generator.add(Activation('relu'))
    Generator.add(Reshape((dim, dim, depth)))
    Generator.add(Dropout(dropout))

    # In: dim x dim x depth
    # Out: 2*dim x 2*dim x depth/2

    Generator.add(UpSampling2D())
    Generator.add(Conv2DTranspose(int(depth/2), 11, padding='same'))
    Generator.add(BatchNormalization(momentum=0.9))
    Generator.add(Activation('relu'))

    Generator.add(Conv2DTranspose(int(depth/2), 7, padding='same'))
    Generator.add(BatchNormalization(momentum=0.9))
    Generator.add(Activation('relu'))

    Generator.add(UpSampling2D())

    Generator.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
    Generator.add(BatchNormalization(momentum=0.9))
    Generator.add(Activation('relu'))

    Generator.add(Conv2DTranspose(int(depth/4), 3, padding='same'))
    Generator.add(BatchNormalization(momentum=0.9))
    Generator.add(Activation('relu'))

    # Out: 28 x 28 x 1 grayscale image [0.0,1.0] per pix

    Generator.add(Conv2DTranspose(1, 3, padding='same'))
    Generator.add(Activation('sigmoid'))
    return Generator


image_shape = (64,64,1)
discriminator = default_discriminator(image_shape)
generator = default_generator()
D_comp = Sequential()
D_comp.add(discriminator)
D_comp.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.0002, decay=6e-8),\
    metrics=['accuracy'])


A_comp = Sequential()
A_comp.add(generator)
for layer in discriminator.layers:
    layer.trainble=False
A_comp.add(discriminator)
A_comp.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.0001, decay=3e-8),\
    metrics=['accuracy'])
for layer in discriminator.layers:
    layer.trainable=True

#Gen = default_generator()
def mean_squared_error_tryme(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))

def import_images(PATH):
        all_images =list()
        for root, subdir_list, filenames in os.walk(PATH):
            for file in filenames:
                if '.JPG' in file:
                    print(os.path.join(root,file))
                    pix = cv2.imread(os.path.join(root,file), cv2.IMREAD_GRAYSCALE)
                    small_image = cv2.resize(pix, (64,64))
                    small_image = cv2.bitwise_not(small_image)
                    small_image = np.asarray(small_image)
                    #small_image = small_image.flatten()
                    small_image = small_image/255.0
                    small_image = small_image.reshape(64,64,1)
                    all_images.append(small_image)
        all_images = np.asarray(all_images)
        return all_images

def plot_images(save2file=False, fake=True, samples=16, noise=None, step=0):
        filename = 'picmin.png'
        if noise is None:
            noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
        else:
            filename = "picmin_%d.png" % step
            filename2 = "inputs_%d.png" % step
        images = generator.predict(noise)


        plt.figure(figsize=(10,10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [64,64])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()


        #z = [[0 1]]


def plot_images2(save2file=False, fake=True, samples=16, noise=None, step=0):
        tryme = Sequential()
        tryme.add(Dense(512, input_shape=(256,), activation='relu'))
        tryme.add(Dropout(.5))
        tryme.add(Activation('relu'))
        tryme.add(Dense(512))
        tryme.add(Dropout(.5))
        tryme.add(Activation('relu'))
        tryme.add(Dense(100))
        for layer in generator.layers:
            layer.trainable = False
        generator.layers.pop()
        tryme.add(generator)
        tryme.add(Activation('relu'))
        tryme.add(Flatten())

        weight_path = 'low_loss.hdf5'
        checkpoint = ModelCheckpoint('low_loss.hdf5', monitor='loss', save_best_only= True, verbose = 0, mode ='max')
        tryme.compile(loss=mean_squared_error_tryme,
                      optimizer=optimizers.SGD(lr=.01),
                      metrics=['mean_squared_error'])

        tryme.summary()

        inputs = ['input1.JPG','input2.JPG','input3.JPG','input4.JPG','input5.JPG','input6.JPG','input7.JPG','input8.JPG']
        inpath ='moredata'
        #filename2 ='tryme'+str(e)+'.png'
        for e,input in enumerate(inputs):
            image_path = os.path.join(inpath,input)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (64,64), interpolation=cv2.INTER_AREA)
            ret,image = cv2.threshold(image,127,255,cv2.THRESH_BINARY_INV)
            image = image/255.0
            image = np.reshape(image, (1,64*64))

            t = np.random.uniform(-1.0, 1.0, size=[1, 256])

            original_weights = tryme.get_weights()
            history = tryme.fit(t, image, verbose=0, epochs=100,  callbacks = [checkpoint])

            tryme.load_weights(weight_path)
            c = tryme.predict(t)

            tryme.set_weights(original_weights)
            out_image = c[0]
            out_image = np.reshape(out_image, [64, 64])
            ori_image = np.reshape(image, [64,64])
            plt.subplot(4, 4, e+1)
            plt.axis('off')
            plt.imshow(ori_image, cmap='gray')
            plt.subplot(4, 4,(e+9))
            plt.axis('off')
            plt.imshow(out_image, cmap='gray')
        plt.tight_layout()
        plt.savefig("final_out2.png")




def train(train_steps=2000, batch_size=256, save_interval=0):
    noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
    x_train = import_images('pics')
    roll_count = 1
    incrementer = 20
    for i in range(train_steps):
        images_train = x_train[np.random.randint(0,
            x_train.shape[0], size=batch_size), :, :, :]
        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
        images_fake = generator.predict(noise)
        x = np.concatenate((images_train, images_fake))
        y = np.ones([2*batch_size, 1])
        y[batch_size:, :] = 0
        d_loss = D_comp.train_on_batch(x, y)

        y = np.ones([batch_size, 1])
        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
        a_loss = A_comp.train_on_batch(noise, y)
        log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
        log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
        print(log_mesg)
        if (i)==0:
            plot_images(save2file=True, samples=noise_input.shape[0],\
                noise=noise_input, step=(i+1))
        if save_interval>0:
            if (i+1)%save_interval==0:
                plot_images(save2file=True, samples=noise_input.shape[0],\
                    noise=noise_input, step=(i+1))
        if (i+1) == 30000:
            plot_images2(save2file=True, samples=noise_input.shape[0],\
                noise=noise_input, step=(i+1))





noise = np.random.uniform(-1.0, 1.0, size=[1, 100])
x = generator.predict(noise)
x = x[0]
x = np.reshape(x, [64,64])
plt.imshow(x, cmap='gray')
train(train_steps = 30000, batch_size=32, save_interval=20)
generator.save('generator.h5')
discriminator.save('discriminator.h5')
