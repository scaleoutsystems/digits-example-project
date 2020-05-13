import keras
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from sklearn.metrics import classification_report

import requests

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Create an initial CNN Model
def create_seed_model():
	# input image dimensions
	img_rows, img_cols = 28, 28
	input_shape = (img_rows, img_cols, 1)
	num_classes = 10

	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))

	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))

	model.compile(loss=keras.losses.categorical_crossentropy,
    	          optimizer=keras.optimizers.Adadelta(),
        	      metrics=['accuracy'])
	return model

if __name__ == '__main__':

	model = create_seed_model()
	

