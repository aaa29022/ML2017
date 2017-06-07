import sys
import numpy as np
from utils import *
from keras.models import Sequential, load_model, Model
from keras.layers import Embedding, Reshape, Merge, Dropout, Dense, Input, merge, Add
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint

import json

import tensorflow as tf
import keras.backend as K 
from keras.backend.tensorflow_backend import set_session

gpu_options = tf.GPUOptions(allow_growth = True)
config = tf.ConfigProto(gpu_options = gpu_options)
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config = config))

k_factors = int(sys.argv[1])
model_option = sys.argv[2]
nb_epochs = 500

def main():
    train, n_users, m_items = load_train('train.csv')
    train, valid = split_valid(train, 0.1)

    if model_option == 'MF':
        model = mf_model(n_users, m_items, k_factors)
    elif model_option == 'NN':
        model = dnn_model(n_users, m_items, k_factors)

    earlystop = EarlyStopping(monitor = 'val_loss', 
                              patience = 3, 
                              verbose = 1) 

    checkpoint = ModelCheckpoint(filepath = 'hw6_best.hdf5',
                                 verbose = 1,
                                 save_best_only = True,
                                 save_weights_only = True,
                                 monitor = 'val_loss',
                                 mode = 'min')
    
    history = model.fit([train[:, 0], train[:, 1]], 
                        train[:, 2], 
                        validation_data = ([valid[:, 0], valid[:, 1]], valid[:, 2]),
                        epochs = nb_epochs, 
                        batch_size = 32,
                        callbacks = [earlystop, checkpoint])

    valid_loss = np.array(history.history['val_loss'])
    print ('min validation loss:', valid_loss.min(axis = 0))    

if __name__ == '__main__':
    main()
