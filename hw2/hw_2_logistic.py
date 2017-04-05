# -*- coding: utf-8 -*-
import sys
import math
import numpy as np

dim = 212

x_data = np.genfromtxt(sys.argv[1], delimiter = ",")
x_data = np.delete(x_data, 0 ,0)
x_data = np.append(x_data, x_data ** 2, axis = 1)

x_test = np.genfromtxt(sys.argv[3], delimiter = ",")
x_test = np.delete(x_test, 0 ,0)
x_test = np.append(x_test, x_test ** 2, axis = 1)

y_data = np.genfromtxt(sys.argv[2])
y_data = y_data.reshape(32561, 1)

mean = np.reshape(np.mean(np.append(x_data, x_test, axis = 0), axis = 0), (1, dim))
std = np.reshape(np.std(np.append(x_data, x_test, axis = 0), axis = 0), (1, dim))
x_data = (x_data - mean) / std
x_test = (x_test - mean) / std
# print (x_data.shape, y_data.shape, mean.shape, std.shape)
# np.savetxt("x_data.csv", x_data, fmt = '%1.4f', delimiter = ",")

b = 1.0 # initial b
w = np.ones((dim, 1)) # initial w
lr = 0.8 # learning rate
iteration = 30000

b_lr = 0.0
w_lr = np.zeros((dim, 1))
lamda = 0

# Iterations
for k in range(iteration):
    # if k % 1000 == 999:
    #     print ("round", k + 1)
    
    z = np.dot(x_data, w) + b
    z = np.clip(z, -709, 709)

    sigmoid = 1 / (1 + np.exp(-z))
    tmp = sigmoid - y_data
    b_grad = np.sum(tmp)
    w_grad = np.dot(tmp.T, x_data).T + lamda * w 
        
    b_lr = b_lr + b_grad ** 2
    w_lr = w_lr + w_grad ** 2
        
    # Update parameters.
    b = b - lr/np.sqrt(b_lr) * b_grad
    w = w - lr/np.sqrt(w_lr) * w_grad

f = open('w_1.txt', 'w')
print (w, file = f)
f.close()

# print (b)

x_test = np.dot(x_test, w) + b
f = open(sys.argv[4], 'w')
f.write('id,label\n')
for k in range(16281):
    z = x_test[k, 0]
    z = np.clip(z, -709, 709)

    if (1 / (1 + math.exp(-z))) > 0.5:
        out = 1
    else:
        out = 0
    f.write("{},{}\n".format(k + 1, out))
f.close()
