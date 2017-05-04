import gc
import sys
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
    model.add(Conv2D(32, (3, 3), input_shape = (48, 48, 1)))
    model.add(Activation('elu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(32, (3, 3), padding = 'same'))
    model.add(Activation('elu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding = 'same'))
    model.add(Activation('elu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(units = 512, activation = 'elu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(units = 7, activation = 'softmax'))

    model.compile(loss = 'categorical_crossentropy', optimizer = "adam", metrics = ['accuracy'])
    model.summary()

    return model

def main():
    x_train, y_train = load_train(sys.argv[1])
    (x_train, y_train), (x_valid, y_valid) = divide_data(x_train, y_train)
    x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
    x_valid = x_valid.reshape(x_valid.shape[0], 48, 48, 1)
    # x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)
    print (x_train.shape, y_train.shape, x_valid.shape, y_valid.shape)

    for count in range(1):
        train_loss, train_acc, valid_loss, valid_acc = [], [], [], []

        datagen = ImageDataGenerator(rotation_range = 10, width_shift_range = 0.1
                                     , height_shift_range = 0.1, horizontal_flip = True)
        datagen.fit(x_train)
        model = build_model()
        earlystop = EarlyStopping(monitor = 'val_loss', min_delta = 0.00001
                                     , patience = 5, verbose = 1, mode = 'auto')
        history = model.fit_generator(datagen.flow(x_train, y_train, batch_size = 32)
                                     , steps_per_epoch = x_train.shape[0], callbacks = [earlystop]
                                     , validation_data = (x_valid, y_valid),
                                      epochs = 25)

        score = model.evaluate(x_train,y_train)
        print ('\nTrain Loss:', score[0], 'Train Acc:', score[1])
        train_loss.append(score[0])
        train_acc.append(score[1])

        score = model.evaluate(x_valid, y_valid)
        print ('\nValidation Loss:', score[0], 'Validation Acc:', score[1])
        valid_loss.append(score[0])
        valid_acc.append(score[1])

        filename = 'model_cnn.h5'
        model.save(filename)
        dump_history(history, 'result_cnn.csv')

        gc.collect()

if __name__ == '__main__':
    main()

