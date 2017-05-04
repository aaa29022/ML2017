import gc
import sys
import numpy as np
import pandas as pd
from utils import *
from keras.models import load_model

def main():
    x_test = load_test(sys.argv[1])
    x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)

    model = load_model('./model_cnn.h5')

    y_test = model.predict(x_test)
    y_test = np.argmax(y_test, axis = 1)
    save_prediction(y_test, sys.argv[2])

    gc.collect()

if __name__ == "__main__":
    main()