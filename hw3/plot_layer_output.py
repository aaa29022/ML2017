# -- coding: utf-8 --
import os
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
import numpy as np
from utils import load_train

def main():
    filter_dir = './img/'
    if not os.path.isdir(filter_dir):
        os.mkdir(filter_dir)

    filter_dir = './img/filter/'
    if not os.path.isdir(filter_dir):
        os.mkdir(filter_dir)

    emotion_classifier = load_model('./model_cnn.h5')
    layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers[1:])

    input_img = emotion_classifier.input
    name_ls = ['activation_1']
    collect_layers = [ K.function([input_img, K.learning_phase()], [layer_dict[name].output]) for name in name_ls ]

    x_train, y_train = load_train('train.csv')
    x_train = x_train.reshape(x_train.shape[0], 1, 48, 48, 1)

    choose_id = 2044
    photo = x_train[choose_id]
    for cnt, fn in enumerate(collect_layers):
        im = fn([photo, 0]) #get the output of that layer
        fig = plt.figure(figsize=(14, 8))
        nb_filter = im[0].shape[3]
        for i in range(nb_filter):
            ax = fig.add_subplot(nb_filter/16, 16, i+1)
            ax.imshow(im[0][0, :, :, i], cmap='YlGnBu')
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            plt.tight_layout()
        fig.suptitle('Output of layer{} (Given image{})'.format(cnt, choose_id))
        img_path = filter_dir
        if not os.path.isdir(img_path):
            os.mkdir(img_path)
        fig.savefig(os.path.join(img_path,'layer{}'.format(cnt)))

if __name__ == "__main__":
    main()