from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from gensim.models import Word2Vec
import numpy as np
import pickle
import gc
from input_handler import *
from model import SiameseBiLSTM
from config import *
import string
import re
from keras.utils.np_utils import to_categorical
from keras.models import load_model
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
import matplotlib.pyplot as plt

"""

Workspace for testing functions 

"""

def plot_model_p():

    (f, (ax1, ax2)) = plt.subplots(1, 2)  # , sharey=True)
    ax1.plot(history['acc'], 'r-')
    ax1.plot(history['val_acc'])
    ax1.set_title('model accuracy')
    ax1.set_ylabel('error %')
    ax1.set_xlabel('epoch')
    ax1.legend(['train', 'test'], loc='lower right')
    ax2.plot(history['loss'], 'r-')
    ax2.plot(history['val_loss'])
    ax2.set_title('model loss')
    ax2.set_ylabel('loss')
    ax2.set_xlabel('epoch')
    ax2.legend(['train', 'test'], loc='upper right')

    # f.set_figwidth(30)
    # ax2.set_figwidth(30)

    plt.rcParams['figure.figsize'] = [16, 9]
    f.set_size_inches((12, 5))
    plt.show()





def count_entailment():
    train_data, test_data, trial_data = load_SICK()
    ent_count = 0
    neu_count = 0
    con_count = 0

    for row in train_data:
        if row[3] == CONTRADICTION:
            con_count += 1
        elif row[3] == NEUTRAL:
            neu_count += 1
        elif row[3] == ENTAILMENT:
            ent_count += 1

    for row in test_data:
        if row[3] == CONTRADICTION:
            con_count += 1
        elif row[3] == NEUTRAL:
            neu_count += 1
        elif row[3] == ENTAILMENT:
            ent_count += 1

    for row in trial_data:
        if row[3] == CONTRADICTION:
            con_count += 1
        elif row[3] == NEUTRAL:
            neu_count += 1
        elif row[3] == ENTAILMENT:
            ent_count += 1

    print('contradictions:{}\nneutral:{}\nentailment:{}'.format(con_count,neu_count,ent_count))
    print('total:{}'.format(con_count+neu_count+ent_count))
    print('train:{}\ntest:{}\ntrial:{}'.format(len(train_data),len(test_data),len(trial_data)))
    


def one_hot(a):
    b = np.zeros_like(a)
    b[np.where(a == np.max(a))] = 1
    return b

def one_hot_practice():

    # four different gates

    training_data = np.array([[0,0],[0,1], [1,0],[1,1]], "float32")
    train_labels = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    model = Sequential()
    model.add(Dense(128, input_dim=2, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(4, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(training_data, train_labels, nb_epoch=500, verbose=2)

    print model.predict(training_data).round()


def remove_punct(sentences):
        for x in range(0, len(sentences)):
            sentences[x] = sentences[x].replace(',','')
        return sentences

def data_prep():
    filters = '!".#$%&()*+/,:;-<=>?@[\\]^_`{|}~\t\n'  # Allow '-'
    s = 'a girl from asia, in front of a window made of bricks, looks surprised'
    s = s.replace(',','')
    print(s)

    # Load SICK training data
    train_data, test_data, trial_data = load_SICK()
    sentences1 = []
    sentences2 = []

    for x in range(0, len(train_data)):
        sentences1.append(train_data[x][1])
        sentences2.append(train_data[x][2])

    # Build Tokenizer
    docs = sentences1 + sentences2
    docs = [string.lower() for string in docs]
    remove_punct(docs)

    print(docs)


    t = Tokenizer(filters=filters) 
    t.fit_on_texts(docs)
    word_index = t.word_index
    #print(word_index.items())

    # Split the sentences into lists of words
    docs = [string.split() for string in docs]

    #print(docs)
    #Vectorize all words in docs
    model = Word2Vec(docs, min_count=1, size=50)
    word_vector = model.wv
    word_vector.get_keras_embedding(True)

    # Find word most similar to man
    result = word_vector.most_similar(positive=['asia'])
    print("{}: {:.4f}".format(*result[0]))
    #print(word_vector['knife'])


if __name__ == '__main__':
    count_entailment()