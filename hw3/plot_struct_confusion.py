import sys
import itertools
from utils import *
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.models import load_model
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):
    """
    This function prints and plots the confusion matrix.
    """
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def main():
    x_train, y_train = load_train('train.csv')
    (x_train, y_train), (x_valid, y_valid) = divide_data(x_train, y_train)
    x_valid = x_valid.reshape(x_valid.shape[0], 48, 48, 1)
    print (x_valid.shape)

    model = load_model(sys.argv[1])
    model.summary()
    plot_model(model, to_file=sys.argv[2])

    predictions = model.predict_classes(x_valid)
    y_valid = np.array(y_valid).argmax(1)
    conf_mat = confusion_matrix(y_valid, predictions)

    plt.figure()
    plot_confusion_matrix(conf_mat, classes=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"])
    fig = plt.gcf()
    fig.savefig(sys.argv[3])

if __name__ == '__main__' :
    main()