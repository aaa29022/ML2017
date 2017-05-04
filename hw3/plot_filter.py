import os
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
from tqdm import *
import numpy as np

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-7)

def grad_ascent(num_step, input_image_data, iter_func):
    step = 0.015
    for i in range(num_step):
        loss_value, grads_value = iter_func([input_image_data])
        input_image_data += grads_value * step
    return [input_image_data, loss_value]

def get_num_filters(layer):
    if K.ndim(layer) == 2:
        return layer.shape[1]
    else:
        if K.image_data_format() == 'channels_first':
            return K.int_shape(layer)[1]
        else:
            return K.int_shape(layer)[3]

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
    collect_layers = [ layer_dict[name].output for name in name_ls ]

    NUM_STEPS = 240
    RECORD_FREQ = 40

    for cnt, c in enumerate(collect_layers):

        nb_filter = get_num_filters(c)
        filter_imgs = [[] for i in range(NUM_STEPS//RECORD_FREQ)]
        for filter_idx in range(nb_filter):
            input_img_data = np.random.random((1, 48, 48, 1)) # random noise
            target = K.mean(c[:, :, :, filter_idx])
            grads = normalize(K.gradients(target, input_img)[0])
            iterate = K.function([input_img], [target, grads])
            
            for step_idx in range(NUM_STEPS//RECORD_FREQ):
                [input_img_data, loss] = grad_ascent(RECORD_FREQ, input_img_data, iterate)
                filter_imgs[step_idx].append([np.array(input_img_data.reshape(48, 48)), loss])

        for it in tqdm(range(NUM_STEPS//RECORD_FREQ)):
            fig = plt.figure(figsize=(14, 10))
            for i in range(nb_filter):
                ax = fig.add_subplot(nb_filter/8, 8, i+1)
                ax.imshow(filter_imgs[it][i][0], cmap='YlGnBu')
                plt.xticks(np.array([]))
                plt.yticks(np.array([]))
                plt.xlabel('{:.3f}'.format(filter_imgs[it][i][1]))
                plt.tight_layout()
            fig.suptitle('Filters of layer {} (# Ascent Epoch {} )'.format(name_ls[cnt], it*RECORD_FREQ))
            img_path = os.path.join(filter_dir, '{}'.format(name_ls[cnt]))
            if not os.path.isdir(img_path):
                os.mkdir(img_path)
            fig.savefig(os.path.join(img_path,'e{}'.format(it*RECORD_FREQ)))

if __name__ == "__main__":
    main()