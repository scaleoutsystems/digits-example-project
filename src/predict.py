import numpy as np
from PIL import Image

class model:
    
    def __init__(self):
        self.model = None
        
    def predict(self, inp):
        if 'json' in inp:
            pred = self.model.predict(np.array(inp['json']))
            return np.array(np.argmax(pred[0]))
        if 'file' in inp:
            print('file')
            filename = inp['file']
            print(filename)
            print('loading image')
            try:
                img_array = np.asarray(Image.open(filename))
            except:
                print('failed to load image.')
            print('loaded image')
            print('image shape')
            print(img_array.shape)
            if img_array.shape == (28,28):
                img_array = np.expand_dims(img_array, 2)
            else:
                print('Image has wrong dimension')
            print('calling predict')
            pred = self.model.predict(np.array([img_array]))
            return np.array(np.argmax(pred[0]))

