from __future__ import print_function
import sys
import keras
import tensorflow as tf 
import pickle
tf.logging.set_verbosity(tf.logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from read_data import read_training_data
#from scaleout.alliance.runtime.runtimeclient import RuntimeClient


def train(model):
    print("-- RUNNING TRAINING --")
    """ Respond to a training request:
        Download the global model and train on the new local data. """
        
    # TODO - move the communication to the MemberRunTimeClient ??
    #client = RuntimeClient()

    # Fetch global model for project
   # print("inited client")
   # global_model_candidate = client.get_global_model()
   # print("got the global model")

    batch_size = 32
    num_classes = 10
    epochs = 1

    # Input image dimensions
    img_rows, img_cols = 28, 28

    # The data, split between train and test sets
    (x_train, y_train, classes) = read_training_data(sample_fraction=0.01)
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_train = x_train.astype('float32')
    x_train /= 255
    y_train = keras.utils.to_categorical(y_train, num_classes)

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)

    # Submit the candidate model
    #client.send_model(global_model_candidate)

    print("-- TRAINING COMPLETED --")
    return model

if __name__ == '__main__':
    # Read the model
    print(sys.argv[1],sys.argv[2])
    with open(sys.argv[1],"rb") as fh:
        model = pickle.loads(fh.read())
    model = train(model)
    with open(sys.argv[2],"wb") as fh:
        model = fh.write(pickle.dumps(model))

