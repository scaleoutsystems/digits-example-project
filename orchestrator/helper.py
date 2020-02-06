import keras
from scaleout.runtime.runtime import Runtime
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from sklearn.metrics import classification_report
from read_data import read_test_data

import requests

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Create an initial global CNN Model

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

def generate_report(model):
	img_rows, img_cols = 28, 28
	x_test, y_test, classes = read_test_data()
	x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
	x_test = x_test.astype('float32')
	x_test /= 255
	y_predict = model.predict_classes(x_test)
	report = classification_report(y_test, y_predict)
	return report
    
def send_report(url, report):
	r = requests.post(url,report)
	return(r.status_code)

if __name__ == '__main__':

	model = create_seed_model()
	