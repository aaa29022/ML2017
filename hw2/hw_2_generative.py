# -*- coding: utf-8 -*-
import sys
import math
import numpy as np
from numpy.linalg import inv

x_data = np.genfromtxt(sys.argv[1], delimiter = ",")
x_data = np.delete(x_data, 0 ,0)
y_data = np.genfromtxt(sys.argv[2])
y_data = y_data.reshape(32561, 1)
x_data = np.append(x_data, y_data, axis = 1)

c1 = x_data[x_data[:, 106] != 0]
c2 = x_data[x_data[:, 106] == 0]
c1 = np.delete(c1, 106, axis = 1)
c2 = np.delete(c2, 106, axis = 1)

u1 = np.reshape(np.mean(c1, axis = 0), (106, 1))
u2 = np.reshape(np.mean(c2, axis = 0), (106, 1))
N1 = c1.shape[0]
N2 = c2.shape[0]

cov1 = np.zeros((106, 106))
for i in range(N1):
	cov1 = cov1 + (c1[i] - u1.T).T * (c1[i] - u1.T)
cov1 = cov1 / N1

cov2 = np.zeros((106, 106))
for i in range(N2):
	cov2 = cov2 + (c2[i] - u2.T).T * (c2[i] - u2.T)
cov2 = cov2 / N2

cov_inv = inv((N1 / (N1 + N2)) * cov1 + (N2 / (N1 + N2)) * cov2)
w = np.dot((u1 - u2).T, cov_inv).T
b = (-0.5) * np.dot(np.dot(u1.T, cov_inv), u1) + 0.5 * np.dot(np.dot(u2.T, cov_inv), u2) + math.log(N1 / N2)
# print (u1.shape, u2.shape, cov_inv.shape, w.shape, b)

z_data = np.genfromtxt(sys.argv[3], delimiter = ",")
z_data = np.delete(z_data, 0 ,0)
z_data = np.dot(z_data, w) + b

f = open(sys.argv[4], 'w')
f.write('id,label\n')
for k in range(16281):
	if (1 / (1 + math.exp((-1) * z_data[k, 0]))) > 0.5:
		out = 1
	else:
		out = 0
	f.write("{},{}\n".format(k + 1, out))
