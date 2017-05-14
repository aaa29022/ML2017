import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def main():
    data = np.load(sys.argv[1])

    dataset_num = 200
    set_var = np.zeros((dataset_num, ))
    for i in range(dataset_num):
        set_var[i] = np.var(data[str(i)].flatten())
    set_var = set_var.reshape(dataset_num, 1)
    km = KMeans(n_clusters = 60, random_state=0).fit(set_var)

    center = np.zeros((60, ))
    for i in range(60):
        cluster = np.argwhere(km.labels_ == i)
        center[i] = set_var[cluster].mean()
    center_index = np.argsort(center)

    dim = np.zeros((dataset_num, ))
    for i in range(dataset_num):
        dim[i] = np.log(np.argwhere(center_index == km.labels_[i]) + 1)

    f = open(sys.argv[2], 'w')
    f.write('SetId,LogDim\n')
    for i in range(dataset_num):
        f.write('{},{}\n'.format(i, dim[i]))
    f.close()

if __name__ == '__main__':
    main()