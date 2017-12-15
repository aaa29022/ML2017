import gc
import sys
import numpy as np
import pandas as pd
import xgboost as xgb

train_X_file = '../data/train_X.csv'
train_Y_file = '../data/train_Y.csv'
test_X_file = '../data/test_X.csv'
test_Y_file = 'output.csv'

def load_data():
    train_X = np.genfromtxt(train_X_file, delimiter = ',')
    sj_X, iq_X = train_X[1:937, 4:], train_X[937:, 4:]

    train_Y = np.genfromtxt(train_Y_file, delimiter = ',')
    sj_Y, iq_Y = train_Y[1:937, 3], train_Y[937:, 3]

    test_X = np.genfromtxt(test_X_file, delimiter = ',')
    sj_test_X, iq_test_X = test_X[1:261, 4:], test_X[261:, 4:]

    return sj_X, sj_Y, iq_X, iq_Y, sj_test_X, iq_test_X

def interpolation(data):
    size = data.shape[0]
    for i in range(data.shape[1]):
        index = np.where(np.isnan(data[:, i]))[0]
        for j in index:
            if j == 0:
                data[j, i] = data[j + 1, i]
            elif j == size - 1 or np.isnan(data[j + 1, i]):
                data[j, i] = data[j - 1, i]
            else:
                data[j, i] = (data[j - 1, i] + data[j + 1, i]) / 2
    return data

def normalize(data):
    mean = data.mean(axis = 0)
    std = data.std(axis = 0)
    data = (data - mean) / std
    return data

def split_valid(data_X, data_Y, n, ratio = 0.1):
    size = int(data_X.shape[0] * ratio)
    index = np.arange(data_X.shape[0])
    np.random.seed(n)
    np.random.shuffle(index)
    data_X = data_X[index]
    data_Y = data_Y[index]
    return data_X[size:], data_X[:size], data_Y[size:], data_Y[:size]

def append_prev_data(data_X, num):
    tmp = np.array(data_X, copy = True)
    tmp = tmp[:, 4:]
    for i in range(num):
        tmp = np.append(tmp[0].reshape(1, tmp.shape[1]), tmp, axis = 0)[:-1]
        data_X = np.append(data_X, tmp, axis = 1)
    return data_X

def append_prev_label(data_X, data_Y, num):
    size = data_X.shape[0]
    tmp = np.array(data_Y, copy = True)
    for i in range(num):
        tmp = np.append(tmp[0], tmp[:-1])
        data_X = np.append(data_X, tmp.reshape(size, 1), axis = 1)
    return data_X

def predict_test_Y(test_X, data_X, data_Y, num, bst):
    size = test_X.shape[1]
    tmp_X = np.array(data_X, copy = True)
    tmp_Y = np.array(data_Y, copy = True)
    test_Y = np.zeros((test_X.shape[0], ))

    for i in range(test_X.shape[0]):
        tmp = np.append(test_X[i], tmp_X)
        tmp = np.append(tmp, tmp_Y)

        dtest = xgb.DMatrix(tmp.reshape(1, tmp.shape[0]))
        test_Y[i] = bst.predict(dtest)[0]
        
        tmp_X = tmp[4:-21]
        tmp_Y = np.append(test_Y[i], tmp_Y)[:5]
    
    return test_Y

def main():
    sj_X, sj_Y, iq_X, iq_Y, sj_test_X, iq_test_X = load_data()
    sj_X, iq_X, sj_test_X, iq_test_X = interpolation(sj_X), interpolation(iq_X), interpolation(sj_test_X), interpolation(iq_test_X)
    sj_X, iq_X, sj_test_X, iq_test_X = normalize(sj_X), normalize(iq_X), normalize(sj_test_X), normalize(iq_test_X)
    
    sj_X = append_prev_data(sj_X, 5)
    iq_X = append_prev_data(iq_X, 5)

    sj_X = append_prev_label(sj_X, sj_Y, 5)
    iq_X = append_prev_label(iq_X, iq_Y, 5)

    sjt_X, sjv_X, sjt_Y, sjv_Y = split_valid(sj_X, sj_Y, n = 4)
    iqt_X, iqv_X, iqt_Y, iqv_Y = split_valid(iq_X, iq_Y, n = 4)

    sj_dtrain = xgb.DMatrix(sjt_X, label = sjt_Y)
    sj_dvalid = xgb.DMatrix(sjv_X, label = sjv_Y)

    iq_dtrain = xgb.DMatrix(iqt_X, label = iqt_Y)
    iq_dvalid = xgb.DMatrix(iqv_X, label = iqv_Y)

    param = {'objective':'reg:linear'}
    param['eval_metric'] = ['mae']
    sj_evallist  = [(sj_dvalid,'eval'), (sj_dtrain,'train')]
    iq_evallist  = [(iq_dvalid,'eval'), (iq_dtrain,'train')]

    num_round = 20
    sj_bst = xgb.train(param, sj_dtrain, num_round, sj_evallist)
    print()
    iq_bst = xgb.train(param, iq_dtrain, num_round, iq_evallist)

    tmp = sj_X[-1, 4:-21]
    sj_test_Y = predict_test_Y(sj_test_X, tmp, sj_Y[-5:][::-1], 5, sj_bst)
    sj_test_Y = sj_test_Y.astype(int)

    tmp = iq_X[-1, 4:-21]
    iq_test_Y = predict_test_Y(iq_test_X, tmp, iq_Y[-5:][::-1], 5, iq_bst)
    iq_test_Y = iq_test_Y.astype(int)

    prdct = np.append(sj_test_Y, iq_test_Y)
    out = pd.read_csv(test_X_file, sep = ',')
    out = out.loc[:, 'city':'weekofyear']
    out['total_cases'] = prdct.tolist()
    out.to_csv(test_Y_file, sep = ',', index = False)

if __name__ == '__main__':
    main()