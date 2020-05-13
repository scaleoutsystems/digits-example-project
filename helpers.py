import tensorflow as tf
import mnist

def load_model():
    model = tf.keras.models.load_model('digits-clf_v1.h5')
    mn = mnist.model()
    mn.model = model
    return mn
