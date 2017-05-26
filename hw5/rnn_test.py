import sys
import string
import pickle
import numpy as np
import keras.backend as K
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

test_path = sys.argv[1]
output_path = sys.argv[2]
model_path = './rnn_best.hdf5'
pickle_path = './rnn_pickle.p'

num_words = 51867
embedding_dim = 300
max_article_length = 306

def read_data(path,training):
    print ('Reading data from ',path)
    with open(path,'r',encoding='utf8') as f:
    
        tags = []
        articles = []
        tags_list = []
        
        f.readline()
        for line in f:
            if training :
                start = line.find('\"')
                end = line.find('\"',start+1)
                tag = line[start+1:end].split(' ')
                article = line[end+2:]
                
                for t in tag :
                    if t not in tags_list:
                        tags_list.append(t)
               
                tags.append(tag)
            else:
                start = line.find(',')
                article = line[start+1:]
            
            articles.append(article)
            
        if training :
            assert len(tags_list) == 38,(len(tags_list))
            assert len(tags) == len(articles)
    return (tags,articles,tags_list)

def f1_score(y_true,y_pred):
    thresh = 0.4
    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    tp = K.sum(y_true * y_pred,axis=-1)
    
    precision=tp/(K.sum(y_pred,axis=-1)+K.epsilon())
    recall=tp/(K.sum(y_true,axis=-1)+K.epsilon())
    return K.mean(2*((precision*recall)/(precision+recall+K.epsilon())))

def main():

    (_, X_test, tag_list) = read_data(test_path, False)
    myp = pickle.load(open(pickle_path, 'rb'))

    tag_list = myp[0]
    tokenizer = myp[1]
    test_sequences = tokenizer.texts_to_sequences(X_test)
    test_sequences = pad_sequences(test_sequences, maxlen = max_article_length)

    model = load_model(model_path, custom_objects = {'f1_score':f1_score})

    Y_pred = model.predict(test_sequences)
    thresh = 0.4
    with open(output_path,'w',encoding='utf8') as output:
        print ('\"id\",\"tags\"',file=output)
        Y_pred_thresh = (Y_pred > thresh).astype('int')
        for index,labels in enumerate(Y_pred_thresh):
            labels = [tag_list[i] for i,value in enumerate(labels) if value==1 ]
            labels_original = ' '.join(labels)
            print ('\"%d\",\"%s\"'%(index,labels_original),file=output)

if __name__=='__main__':
    main()
