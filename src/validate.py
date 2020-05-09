import sys
import tensorflow as tf 
tf.logging.set_verbosity(tf.logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import keras
from .read_data import read_training_data
import pickle
import json

def validate(model):
    
    print("-- RUNNING VALIDATION! --")

    try:
        x_test, y_test, classes = read_training_data(sample_fraction=0.01)
        # Input image dimensions
        img_rows, img_cols = 28, 28
        num_classes = 10
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        x_test = x_test.astype('float32')
        x_test /= 255
        y_test = keras.utils.to_categorical(y_test, num_classes)
        model_score = model.evaluate(x_test, y_test, verbose=0)
        print('Training loss:', model_score[0])
        print('Training accuracy:', model_score[1])
    except Exception as e:
        print ("failed to run scoring {}".format(e))
        exit(-1)

    report = { 
                "training_loss": model_score[0],
                "training_accuracy": model_score[1]
            }

    print("-- VALIDATION COMPLETED --")
    return report

if __name__ == '__main__':
    # Read the model
    print(sys.argv[1],sys.argv[2])
    with open(sys.argv[1],"rb") as fh:
        model = pickle.loads(fh.read())
    report = validate(model)

    with open(sys.argv[2],"w") as fh:
        fh.write(json.dumps(report))
