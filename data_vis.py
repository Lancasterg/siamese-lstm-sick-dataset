from load_sick import *
from inputHandler import *
from model import SiameseBiLSTM
from inputHandler import word_embed_meta_data, create_test_data
from config import siamese_config
from load_sick import *
from operator import itemgetter
from keras.models import load_model
import pandas as pd
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
from keras.utils.np_utils import to_categorical
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools


def one_hot(a):
    b = np.zeros_like(a)
    b[np.where(a == np.max(a))] = 1
    return b

def prepare_data(extend = False):
    sentences1, sentences2, is_similar = load_sick_entailment(mode=TRAIN)
    test_sentences1, test_sentences2, test_labels = load_sick_entailment(mode=TEST)

    if extend ==  True:
        sentences1, sentences2, is_similar, test_sentences1, test_sentences2, test_labels = extend_train_set(sentences1, sentences2, is_similar, test_sentences1, test_sentences2, test_labels)
    
    sentences1 = remove_punct(sentences1)
    sentences2 = remove_punct(sentences2)

    test_sentences1 = remove_punct(test_sentences1)
    test_sentences2 = remove_punct(test_sentences2)

    is_similar = to_categorical(is_similar, num_classes=3)
    test_labels = to_categorical(test_labels, num_classes=3)

    # creating word embedding meta data for word embedding 
    tokenizer, embedding_matrix = word_embed_meta_data(sentences1 + sentences2 ,  siamese_config['EMBEDDING_DIM'])

    test_sentence_pairs = [(x1, x2) for x1, x2 in zip(test_sentences1, test_sentences2)]

    test_data_x1, test_data_x2, leaks_test = create_test_data(tokenizer, test_sentence_pairs,  siamese_config['MAX_SEQUENCE_LENGTH'])

    return(test_data_x1, test_data_x2, leaks_test, test_labels)



def create_confusion_matrix(model, data):
    ''' Create a confusion confusion matrix.
        Tests a network model against test data to generate a
        confusion matrix.
        Args:
            model: the neural network model
            test_img: the test images
            test_label: the test labels for the images
    '''
    (test_data_x1, test_data_x2, leaks_test, test_labels) = data

    history = model.predict([test_data_x1, test_data_x2, leaks_test])

    test = model.evaluate([test_data_x1, test_data_x2, leaks_test], test_labels)

    #print (test[0], 100 - test[1] * 100)
    print('\nloss:{}\nacc:{}'.format(test[0],test[1]))
    confusion = np.zeros((len(test_labels[0]), len(test_labels[0])))
    wrong_pos =[]
    for n in range(len(history)):
        enum_history = history[n]
        result = np.where(enum_history == max(enum_history))[0][0]
        actual = np.where(test_labels[n] == max(test_labels[n]))[0][0]
        confusion[actual][result] += 1
        if result != actual:
            wrong_pos.append(n)
    confusion = confusion.astype(int)
    return confusion

def plot_confusion_matrix(cm, classes,normalize=False,
                            title='Confusion matrix',
                            cmap=plt.cm.RdYlGn):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Args:
        cm: the confusion matrix
        classes: the mapping for the predictions
        normalize: normalise the confusion matrix values
        title: title of the plot
        cmap: colours of the matrix
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        cm = cm.astype('int')
        print ('Normalized confusion matrix')
    else:
        print ('Confusion matrix, without normalization')
    

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    miss_list = [0] * len(classes)
    for (i, j) in itertools.product(range(cm.shape[0]),
                                    range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="black" if cm[i, j] > thresh else "black")
        if i != j:
            miss_list[i] += cm[i][j]
    print ('Incorrect classifications: ')
    print (sum(miss_list))

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')




if __name__ == '__main__':
    path_1 = './checkpoints/1534319754/lstm_50_50_0.17_0.25.h5' 
    path_2 = './checkpoints/1534321589/lstm_75_75_0.17_0.25.h5' # best
    path_3 = './checkpoints/1534323046/lstm_128_64_0.17_0.25.h5'
    path_4 = './checkpoints/1534324267/lstm_64_128_0.17_0.25.h5'
    path_5 = './checkpoints/1534326184/lstm_128_128_0.25_0.50.h5'
    path_6 = './checkpoints/1534327409/lstm_75_75_0.25_0.50.h5'
    path_7_ext = './checkpoints/1534329289/lstm_75_75_0.25_0.50.h5'
    path_8_ext = './checkpoints/1534331633/lstm_75_75_0.17_0.25.h5'
    path_9_ext = './checkpoints/1534333387/lstm_75_75_0.17_0.25.h5' # best_ext
    path_10_ext = './checkpoints/1534334972/lstm_75_75_0.17_0.25.h5'
    path_11_ext = './checkpoints/1534337241/lstm_75_75_0.17_0.25.h5'
    classes = ['Contradiction','Neutral', 'Entailment']



    model = load_model(path_2)
    data = prepare_data(extend=False)
    confusion = create_confusion_matrix(model, data)
    plot_confusion_matrix(confusion,classes, normalize=True)
    plt.show()