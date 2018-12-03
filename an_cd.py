import keras
from keras import applications
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Flatten,Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
import os

def AlexNet(width, height, depth, classes):
	model = Sequential()
	model.add(Conv2D(48,
	 kernel_size = (11,11),
	 strides =(2,2),
	 activation = 'relu',
	 input_shape=(width,height,depth),
	))
    	
	model.add(MaxPooling2D(pool_size=(2,2)))

	model.add(Conv2D(128,
	 kernel_size = (5,5),
	 strides =(1,1),
	 activation = 'relu',
	))
	
	model.add(MaxPooling2D(pool_size=(2,2)))
	
	model.add(Conv2D(192,
	 kernel_size = (3,3),
	 strides =(1,1),
	 activation = 'relu',
	))
	
	model.add(Conv2D(192,
	 kernel_size = (3,3),
	 strides =(1,1),
	 activation = 'relu',
	))

	model.add(Conv2D(128,
	 kernel_size = (3,3),
	 strides =(1,1),
	 activation = 'relu',
	))

	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Flatten())
	model.add(Dropout(0.5))
	model.add(Dense(4096, activation = 'relu'))
	model.add(Dropout(0.5))
	model.add(Dense(4096, activation = 'relu'))
	model.add(Dense(17, activation = 'relu'))
	
	return model
 


data_dir = "flowers17" # change to your directory
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")

batch_size = 64
datagen_train = ImageDataGenerator(rescale=1./255,
		rotation_range = 60,
		horizontal_flip = True,
		vertical_flip = True,
		height_shift_range = .2,
		width_shift_range = .2)
datagen_test = ImageDataGenerator(rescale=1./255)

generator_train = datagen_train.flow_from_directory(directory=train_dir,
                                                    batch_size=batch_size,
                                                    target_size=(227,227),
                                                    shuffle = True,
                                                    class_mode = 'categorical')

generator_test = datagen_test.flow_from_directory(  directory=test_dir,
                                                    batch_size=batch_size,
                                                    target_size=(227,227),
                                                    class_mode = 'categorical',
                                                    shuffle = False)

steps_test = generator_test.n // batch_size
print(steps_test)

epochs = 50
steps_per_epoch = generator_train.n // batch_size
print(steps_per_epoch)

#model = AlexNet(227,227,3,3)
#Instead of using the Alexnet like model above, I will use VGG19 and pretrained weights from imagenet

img_width, img_height = 227, 227
base_model = applications.VGG19(weights="imagenet", include_top=False, input_shape=(img_width, img_height,3))


x = base_model.output
x = Flatten()(x)
x = Dense(256, activation = 'relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(17, activation='softmax')(x)

model = Model(input=base_model.input, output = predictions)

for layer in base_model.layers:
	layer.trainable = False

Adam = keras.optimizers.adam(lr=0.001, amsgrad = True)
model.compile(loss='categorical_crossentropy', optimizer= Adam,\
 metrics=['accuracy'])

model.summary()

history= model.fit_generator(generator_train,
                           epochs=30,
                           steps_per_epoch=steps_per_epoch,
                           validation_data = generator_test,
                           validation_steps = steps_test)


for layer in base_model.layers:
	layer.trainable = True

Adam = keras.optimizers.adam(lr=0.0001, amsgrad = True)
model.compile(loss='categorical_crossentropy', optimizer= Adam,\
 metrics=['accuracy'])

model.summary()

history= model.fit_generator(generator_train,
                           epochs=20,
                           steps_per_epoch=steps_per_epoch,
                           validation_data = generator_test,
                           validation_steps = steps_test)
plt.figure(figsize=[8,6])
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)

plt.figure(figsize=[8,6])
plt.plot(history.history['acc'],'r',linewidth=3.0)
plt.plot(history.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)
