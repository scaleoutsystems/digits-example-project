from __future__ import print_function
import sys
import keras
import tensorflow as tf 
import pickle
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from .read_data import read_data
#from scaleout.alliance.runtime.runtimeclient import RuntimeClient


def train(model,data,sample_fraction):
    print("-- RUNNING TRAINING --")

    batch_size = 32
    num_classes = 10
    epochs = 1

    # Input image dimensions
    img_rows, img_cols = 28, 28

    # The data, split between train and test sets
    (x_train, y_train, classes) = read_data(data,sample_fraction=sample_fraction)
    
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)

    print("-- TRAINING COMPLETED --")
    return model

if __name__ == '__main__':
    # Read the model
    print(sys.argv[1],sys.argv[2])
    with open(sys.argv[1],"rb") as fh:
        model = pickle.loads(fh.read())
    model = train(model)
    with open(sys.argv[2],"wb") as fh:
        fh.write(pickle.dumps(model))


