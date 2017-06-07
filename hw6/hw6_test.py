import sys
import json
import numpy as np
from utils import *
from keras.models import Sequential, load_model, Model
from keras.layers import Embedding, Reshape, Merge, Dropout, Dense, Input, merge, Add
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint

n_users = 6040
m_items = 3952
data_dir = sys.argv[1]
output = sys.argv[2]
model_option = sys.argv[3]

def main():
    test = np.genfromtxt('{}test.csv'.format(data_dir), delimiter = ',')
    test = np.delete(test, 0, axis = 0)
    test[:, 1] = test[:, 1] - 1
    test[:, 2] = test[:, 2] - 1

    if model_option == 'MF':
        k_factors = 5
        model = mf_model(n_users, m_items, k_factors)
        model.load_weights('mf_best.hdf5')
    elif model_option == 'NN':
        k_factors = 120
        model = dnn_model(n_users, m_items, k_factors)
        model.load_weights('dnn_best.hdf5')   

    rate_pred = model.predict([test[:, 1], test[:, 2]])
    f = open(output, 'w')
    f.write('TestDataID,Rating\n')
    for i in range(test.shape[0]):
        f.write('{},{}\n'.format(int(test[i, 0]), rate_pred[i, 0]))
    f.close()

if __name__ == '__main__':
    main()
