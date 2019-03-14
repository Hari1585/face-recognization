# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 16:10:25 2018

@author: Hari
"""


# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 14:05:21 2018

@author: Abhishek
"""
#Import the libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

#Intialize the model
model=Sequential()


# Add Convolution Layer
model.add(Convolution2D(32,(3,3),input_shape=(64,64,3),activation="relu"))


#Add Pooling Layer
model.add(MaxPooling2D(pool_size = (2, 2)))

#Add Flattening Layer
model.add(Flatten())

#Add Hidden Layer
model.add(Dense(init="uniform",activation="relu",output_dim=120))

#Add Output layer
model.add(Dense(init="uniform",activation="sigmoid",output_dim=1))

#Compile the model

model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)



test_datagen = ImageDataGenerator(rescale = 1./255)



x_train = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                     class_mode = 'binary')
x_test = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

print(x_train.class_indices)

model.fit_generator(x_train,
                    steps_per_epoch = 250,
                    epochs = 25,
                    validation_data = x_test,
                    validation_steps = 63)

model.save("mymodel.h5")