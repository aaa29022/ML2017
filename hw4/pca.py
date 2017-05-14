import string
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

def load_img():
    img_dir = './faceExpressionDatabase/'
    data = np.zeros((100, 64 * 64))
    for i in enumerate(string.ascii_uppercase[:10]):
        for j in range(10):
            data[10 * i[0] + j] = misc.imread('{}{}{:02d}.bmp'.format(img_dir, i[1], j)).flatten()
    return data.transpose()

def plot_fig(img, label, row, column, name):
    print ('Plotting {} ...'.format(name))
    fig = plt.figure(figsize = (10, 12))
    for i in range(img.shape[1]):
        ax = fig.add_subplot(column, row, i + 1)
        ax.imshow(img[:, i].reshape(64, 64), cmap = 'gray')
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        plt.xlabel(label[i])
    fig.savefig(name)

def main():
    data = load_img()
    label = ['{}'.format(i) for i in range(100)]
    plot_fig(data, label, 10, 10, 'origin.png')

    meanface = np.reshape(data.mean(axis = 1), (64, 64))
    print ('Plotting meanface ...')
    misc.imsave('mean.bmp', meanface)

    data_c = data - meanface.reshape(64 * 64, 1)

    U, s, V = np.linalg.svd(data_c, full_matrices = False)
    S = np.diag(s)

    label = ['eigenface{}'.format(i) for i in range(9)]
    plot_fig(U[:, :9], label, 3, 3, 'eigenface.png')

    reconstruct = U[:, :5].dot(S[:5, :5].dot(V[:5, :])) + meanface.reshape(64 * 64, 1)
    label = ['{}'.format(i) for i in range(100)]
    plot_fig(reconstruct, label, 10, 10, 'reconstruct.png')

    for k in range(99, 0, -1):
        reconstruct = U[:, :k].dot(S[:k, :k].dot(V[:k, :])) + meanface.reshape(64 * 64, 1)
        rmse = np.sqrt(np.mean((reconstruct - data) ** 2))
        if rmse > 2.55:
            print ('{} eigenvectors, rmse {}'.format(k + 1, rmse_prev))
            print ('{} eigenvectors, rmse {}'.format(k, rmse))
            break
        rmse_prev = rmse


if __name__ == '__main__':
    main()