import numpy as np
import pandas as pd
import gc
import os
import sys
import argparse
from tqdm import *
from keras.models import load_model
import keras.backend as K
import matplotlib.pyplot as plt
from utils import load_train

img_dir = './img'
if not os.path.exists(img_dir):
    os.mkdir(img_dir)
img_dir = './img/saliency_map'
if not os.path.exists(img_dir):
    os.mkdir(img_dir)
cmap_dir = os.path.join(img_dir, 'cmap')
if not os.path.exists(cmap_dir):
    os.mkdir(cmap_dir)
origin_dir = os.path.join(img_dir, 'origin')
if not os.path.exists(origin_dir):
    os.mkdir(origin_dir)
partial_see_dir = os.path.join(img_dir,'partial_see')
if not os.path.exists(partial_see_dir):
    os.mkdir(partial_see_dir)

def main():
    x_train, y_train = load_train('train.csv')
    x_train = x_train[:20, :]
    x_train = x_train.reshape(x_train.shape[0], 1, 48, 48, 1)

    emotion_classifier = load_model('./model_cnn.h5')
    input_img = emotion_classifier.input

    for idx in tqdm(range(20)):
        val_proba = emotion_classifier.predict(x_train[idx])
        pred = val_proba.argmax(axis=-1)
        target = K.mean(emotion_classifier.output[:, pred])
        grads = K.gradients(target, input_img)[0]
        fn = K.function([input_img, K.learning_phase()], [grads])

        grads_value = fn([x_train[idx], 0])
        heatmap = np.array(grads_value).reshape(48, 48)
        s = np.sort(heatmap, axis = None)
        clip_rate = 0.1
        clip_size = int(len(s) * clip_rate)
        heatmap = np.clip(heatmap, s[clip_size], s[len(s) - clip_size])
        heatmap = abs(heatmap - np.mean(heatmap))
        heatmap = (heatmap - np.mean(heatmap))/np.std(heatmap)
        heatmap = (heatmap - heatmap.min())/ heatmap.ptp()

        thres = 0.5
        origin = x_train[idx].reshape(48, 48)*255
        see = x_train[idx].reshape(48, 48)
        see[np.where(heatmap <= thres)] = np.mean(see)
        see *= 255

        plt.figure()
        plt.imshow(heatmap, cmap=plt.cm.jet)
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig(os.path.join(cmap_dir, '{}.png'.format(idx)), dpi=100)

        plt.figure()
        plt.imshow(see, cmap='gray')
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig(os.path.join(partial_see_dir, '{}.png'.format(idx)), dpi=100)

        plt.figure()
        plt.imshow(origin, cmap='gray')
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig(os.path.join(origin_dir, '{}.png'.format(idx)), dpi=100)
        
if __name__ == "__main__":
    main()