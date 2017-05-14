import nltk
import word2vec
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from adjustText import adjust_text

def main():
    vocab_number = 700
    dim = 150

    word2vec.word2vec(
        train='./all.txt',
        output='./all.bin',
        sample=0,
        cbow=1,
        size=dim,
        min_count=50,
        window=2,
        negative=5, 
        hs=1)

    model = word2vec.load('./all.bin')

    vocab = model.vocab[1:vocab_number + 1]
    vec = np.zeros((vocab_number, dim))
    for i, v in enumerate(vocab):
        vec[i] = model[v]

    tsne = TSNE(n_components = 2, random_state=0)
    vec = tsne.fit_transform(vec)

    use_tags = set(['JJ', 'NNP', 'NN', 'NNS'])
    puncts = ["'", '.', ':', ";", ',', "?", "!", u"’"]
        
    plt.figure(figsize = (15, 10))
    texts = []
    for i, label in enumerate(vocab):
        pos = nltk.pos_tag([label])
        if (len(label) > 1 and pos[0][1] in use_tags
                and all(c not in label for c in puncts)):
            x, y = vec[i, :]
            texts.append(plt.text(x, y, label))
            plt.scatter(x, y, marker='o')
            plt.xticks(range(-25, 35, 5))
            plt.yticks(range(-25, 35, 5))

    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='k', lw=0.1))

    plt.savefig('word2vec.png', dpi=600)

if __name__ == '__main__':
    main()