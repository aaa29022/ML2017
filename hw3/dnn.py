import gc
import time
from utils import *
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import regularizers
from keras.utils import np_utils
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.callbacks import History
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import keras.backend.tensorflow_backend as KTF

gpu_options = tf.GPUOptions(allow_growth=True)
KTF.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

# gpu_options = tf.GPUOptions(allow_growth = True)
# config = tf.ConfigProto(gpu_options = gpu_options)
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
# set_session(tf.Session(config = config))

def build_model():
    model = Sequential()
    model.add(Dense(input_dim = 48 * 48, units = 512, activation = 'elu'))
    model.add(Dense(units = 119, activation = 'elu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Dense(units = 72, activation = 'elu'))
    model.add(Dense(units = 72, activation = 'elu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Dense(units = 64, activation = 'elu'))
    model.add(Dense(units = 64, activation = 'elu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Dense(units = 32, activation = 'elu'))
    model.add(Dense(units = 32, activation = 'elu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(units = 7, activation = 'softmax'))

    model.compile(loss = 'categorical_crossentropy', optimizer = "adam", metrics = ['accuracy'])
    model.summary()

    return model

def main():
    x_train, y_train = load_train('train.csv')
    x_test = load_test('test.csv')
    (x_train, y_train), (x_valid, y_valid) = divide_data(x_train, y_train)
    print (x_train.shape, y_train.shape, x_test.shape, x_valid.shape, y_valid.shape)

    model = build_model()
    earlystop = EarlyStopping(monitor = 'val_loss', min_delta = 0.00001, patience = 5, verbose = 1, mode = 'auto')
    history = model.fit(x_train, y_train, batch_size = 32, callbacks =
                        [earlystop], validation_data = (x_valid, y_valid),
                        epochs = 200)

    score = model.evaluate(x_train, y_train)
    print ('\nTrain Loss:', score[0], 'Train Acc:', score[1])
    score = model.evaluate(x_valid, y_valid)
    print ('\nValidation Loss:', score[0], 'Validation Acc:', score[1])

    filename = 'model_dnn_' + str(int(time.time())) + '.h5'
    model.save(filename)

    y_test = model.predict(x_test)
    y_test = np.argmax(y_test, axis = 1)
    save_prediction(y_test, 'dnn_output.csv')

    dump_history(history, 'result_dnn.csv')
    gc.collect()

if __name__ == '__main__':
    main()

