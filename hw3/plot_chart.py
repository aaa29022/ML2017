import sys
import numpy as np
import matplotlib.pyplot as plt

def main():
    result = sys.argv[1]
    chart_title = sys.argv[2]

    data = np.genfromtxt(result, delimiter = ',')
    x = np.arange(1, data.shape[0] + 1)

    plt.plot(x, data[:, 0], label = 'Train')
    plt.plot(x, data[:, 2], label = 'Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xlim(0, data.shape[0] + 1)
    plt.legend()
    plt.title(chart_title + ' Loss')
    plt.grid(linestyle = '-')
    fig = plt.gcf()
    fig.savefig(chart_title + '_loss.png')

    plt.clf()

    plt.plot(x, data[:, 1], label = 'Train')
    plt.plot(x, data[:, 3], label = 'Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xlim(0, data.shape[0] + 1)
    plt.legend()
    plt.title(chart_title + ' Accuracy')
    plt.grid()
    fig = plt.gcf()
    fig.savefig(chart_title + '_acc.png')

    plt.clf()

if __name__ == "__main__":
    main()