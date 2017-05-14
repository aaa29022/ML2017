import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def load_img():
    data = np.zeros((481, 16 * 15))
    for i in range(481):
        data[i] = misc.imresize(misc.imread('./hand/hand.seq{}.png'.format(i + 1)), (16, 15), interp='bilinear').flatten()
    pca = PCA(n_components=100)
    data = pca.fit_transform(data)
    return data

def main():
    hand = load_img().flatten()

    data = np.load('data.npz')
    dataset_num = 201
    set_var = np.zeros((dataset_num, ))
    for i in range(dataset_num - 1):
        set_var[i] = np.var(data[str(i)].flatten())
    set_var[200] = np.var(hand)
    set_var = set_var.reshape(dataset_num, 1)
    km = KMeans(n_clusters = 60, random_state=0).fit(set_var)

    center = np.zeros((60, ))
    for i in range(60):
        cluster = np.argwhere(km.labels_ == i)
        center[i] = set_var[cluster].mean()
    center_index = np.argsort(center)

    dim = np.argwhere(center_index == km.labels_[200]) + 1
    print ('dimension of hand rotation sequence datatset: {}'.format(dim[0, 0]))

if __name__ == '__main__':
    main()