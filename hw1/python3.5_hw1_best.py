# -*- coding: utf-8 -*-
import sys
import numpy as np

data = np.genfromtxt(sys.argv[1], delimiter = ",")
data = data[np.arange(10, 18 * 20 * 12, 18)]
data = data[:, 3:]
data = data.reshape(12, 480)
data = data[2:6, :]

x_data = np.empty((0, 9), float)
y_data = np.empty((0, 1), float)
for i in range(len(data)):
    miss = []
    for j in range(data[i].size):
        if (data[i][j] == -1):
            if (data[i][j] != data[i][j - 1]):
                begin = j
            if (data[i][j] != data[i][j + 1]):
                end = j + 1
                miss.append([begin, end])
    for j in range(len(miss)):
        begin = miss[j][0]
        end = miss[j][1]
        data[i][begin : end] = np.interp(list(range(begin, end)), [begin - 1, end], [data[i][begin - 1], data[i][end]])
    for j in range(471):
        x_data = np.append(x_data, [data[i][j:j + 9]], axis = 0)
        y_data = np.append(y_data, [[data[i][j + 9]]], axis = 0)
# print (x_data.shape, y_data.shape)

# y_data = b + w * x_data 
b = 0.0 # initial b
w = np.zeros((9, 1)) # initial w
lr = 0.5 # learning rate
iteration = 30000

b_lr = 0.0
w_lr = np.zeros((9, 1))
lamda = 10000

# Iterations
for k in range(iteration):
    # if k % 1000 == 999:
    #     print ("round", k + 1)
    
    b_grad = 0.0
    w_grad = np.zeros((9, 1))

    tmp = 2.0 * (y_data - b - np.dot(x_data, w))
    b_grad = b_grad - np.sum(tmp)
    w_grad = w_grad - np.dot(tmp.T, x_data).T# + 2 * lamda * w
    
    b_lr = b_lr + b_grad ** 2
    w_lr = w_lr + w_grad ** 2
    
    # Update parameters.
    b = b - lr/np.sqrt(b_lr) * b_grad
    w = w - lr/np.sqrt(w_lr) * w_grad


test = np.genfromtxt(sys.argv[2], delimiter = ",")
test = test[np.arange(9, 18 * 20 * 12, 18)]
test = test[:, 2:]
output = np.array(b + test.dot(w))

f = open(sys.argv[3], 'w')
f.write('id,value\n')
for k in range(240):
	f.write("id_{},{}\n".format(k, output[k, 0]))