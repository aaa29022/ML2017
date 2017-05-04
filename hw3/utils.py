import gc
import time
import numpy as np
import pandas as pd

def load_train(path):
    print ('Load training data with pandas...')
    train = pd.read_csv(path, sep = ',')
    train['feature'] = train['feature'].str.split(' ')

    print ('Processing x_train...')
    x_train = []
    for i in range(train.shape[0]):
        x_train.append(list(map(int, train['feature'][i])))
    x_train = np.array(x_train)
    x_train = x_train / 255
    x_train = x_train.astype('float32')

    print ('Processing y_train...')
    # y_train = np_utils.to_categorical(train['label'].as_matrix(), 7)
    y_train = np.zeros((train.shape[0], 7))
    y_train[np.arange(train.shape[0]), train['label'].as_matrix()] = 1

    return x_train, y_train

def load_test(path):
    print ('Load testing data with pandas...')
    test = pd.read_csv(path, sep = ',')
    test['feature'] = test['feature'].str.split()

    print ('Processing x_test...')
    x_test = []
    for i in range(test.shape[0]):
        x_test.append(list(map(int, test['feature'][i])))
    x_test = np.array(x_test)
    x_test = x_test / 255
    x_test = x_test.astype('float32')
    return x_test

def divide_data(x_train, y_train, val_proportion = 0.1):
    val_size = int(x_train.shape[0] * val_proportion)
    return (x_train[val_size:], y_train[val_size:]), (x_train[:val_size], y_train[:val_size])

def save_prediction(y_test, filename):
    #filename = 'output_' + str(int(time.time())) + '.csv'
    print ('Saving to file {}...'.format(filename))
    f = open(filename, 'w')
    f.write('id,label\n')
    for i in range(y_test.shape[0]):
        f.write('{},{}\n'.format(i, y_test[i]))
    f.close()

def dump_history(history, filename):
    # filename = 'result_' + str(int(time.time())) + '.csv'
    train_loss = np.array(history.history['loss'])
    train_acc = np.array(history.history['acc'])
    valid_loss = np.array(history.history['val_loss'])
    valid_acc = np.array(history.history['val_acc'])
    with open(filename, 'a') as f:
        for i in range(train_loss.shape[0]):
            f.write('{}, {}, {}, {}\n'.format(train_loss[i], train_acc[i], valid_loss[i], valid_acc[i]))



