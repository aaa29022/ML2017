import numpy as np
from keras.models import Sequential, load_model, Model
from keras.layers import Embedding, Reshape, Merge, Dropout, Dense, Input, merge, Add, Concatenate
from keras.layers.core import Flatten

def load_train(filename):
    print ('Load train.csv')
    data = np.genfromtxt(filename, delimiter = ',')
    data = np.delete(data, 0, axis = 0)
    data = np.delete(data, 0, axis = 1)

    col_max = data.max(axis = 0)

    data[:, 0] = data[:, 0] - 1
    data[:, 1] = data[:, 1] - 1

    return data, int(col_max[0]), int(col_max[1])

def split_valid(data, ratio = 0.1):
    size = int(data.shape[0] * ratio)
    index = np.arange(data.shape[0])
    np.random.seed(0)
    np.random.shuffle(index)
    data = data[index]
    return data[size:], data[:size]

def mf_model(n_users, m_items, k_factors):
    P_input = Input(shape = [1], dtype = 'int32')
    P = Embedding(n_users, k_factors, input_length = 1)(P_input)
    P = Reshape((k_factors,))(P)
    P_bias = Embedding(n_users, 1, input_length = 1)(P_input)
    P_bias = Reshape((1, ))(P_bias)

    Q_input = Input(shape = [1], dtype = 'int32')
    Q = Embedding(m_items, k_factors, input_length = 1)(Q_input)
    Q = Reshape((k_factors,))(Q)
    Q_bias = Embedding(m_items, 1, input_length = 1)(Q_input)
    Q_bias = Reshape((1, ))(Q_bias)

    _output = merge([P, Q], mode = 'dot', dot_axes = 1)
    _output = Add()([_output, P_bias])
    _output = Add()([_output, Q_bias])

    model = Model(input = [P_input, Q_input], output = _output)

    model.compile(loss = 'mse', optimizer = 'adam', metrics = [])
    model.summary()
    return model

def dnn_model(n_users, m_items, k_factors):
    P = Sequential()
    P.add(Embedding(n_users, k_factors, input_length = 1))
    P.add(Reshape((k_factors,)))
    Q = Sequential()
    Q.add(Embedding(m_items, k_factors, input_length = 1))
    Q.add(Reshape((k_factors,)))

    model = Sequential()
    model.add(Merge([P, Q], mode = 'concat'))
    model.add(Dropout(0.3))
    model.add(Dense(k_factors, activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation = 'linear'))

    model.compile(loss = 'mse', optimizer = 'nadam', metrics = [])
    model.summary()

    return model
