from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from gensim.models import Word2Vec
from config import EMBEDDING_DIM
import numpy as np
import pickle
import gc
from numpy import array
import numpy as np

# constants for test / train / trial data
TEST = 'TEST'
TRAIN = 'TRAIN'
TRIAL = 'TRIAL'

# constants for min and max relatedness
MIN = 1
MAX = 5

# constants for entailment
ENTAILMENT = 'ENTAILMENT'
NEUTRAL = 'NEUTRAL'
CONTRADICTION = 'CONTRADICTION'

def load_SICK():
    ''' 
    loads the SICK data set as a tuple.
    Returns:
        train_data (tuple): tuple of the training data sentences and their labels
        test_data (tuple): tuple of the test data sentences and their labels
        trial_data (tuple): tuple of the trial data sentences and their labels
    '''
    f = open('SICK.txt', 'r') # open the file for reading
    train_data = []
    test_data = []
    trial_data = []
    data = []
    for row_num, line in enumerate(f):
        values = line.strip().split('\t')
        if row_num == 0: # first line is the header
             header = values
        else:
            data.append([v for v in values])
    sick = array(data)
    f.close() # close the file

    for row in sick:
        if row[11] == TEST:
            test_data.append(row)
        elif row[11] == TRAIN:
            train_data.append(row)
        elif row[11] == TRIAL:
            train_data.append(row)

    return(train_data, test_data, trial_data)


def load_sick_entailment(mode=TRAIN):
    """
    load one of the data sets from the SICK data set. 
    Uses entailment to train and test
    Args:
        mode (string): TEST or TRAIN to load the training or testing set
    Returns:
        sentences1 (list): First set of senteces
        sentences2 (list): Second set of sentences
        entailment (list): Entailment labels 
    """
    train_data, test_data, trial_data = load_SICK()
    sentences1 = []
    sentences2 = []
    entailment = []
    if mode == TRAIN:
        for x in range(0, len(train_data)):
            sentences1.append(train_data[x][1])
            sentences2.append(train_data[x][2])

            if train_data[x][3] == CONTRADICTION:
                entailment.append(0)
            elif train_data[x][3] == NEUTRAL:
                entailment.append(1)
            elif train_data[x][3] == ENTAILMENT:
                entailment.append(2)
    elif mode==TEST:
        for x in range(0, len(test_data)):
            sentences1.append(test_data[x][1])
            sentences2.append(test_data[x][2])

            if test_data[x][3] == CONTRADICTION:
                entailment.append(0)
            elif test_data[x][3] == NEUTRAL:
                entailment.append(1)
            elif test_data[x][3] == ENTAILMENT:
                entailment.append(2)


    return sentences1, sentences2, entailment


def extend_train_set(sentences1, sentences2, is_similar, test_sentences1, test_sentences2, test_labels):
    """ 
    Increase the size of the training set by adding the first half of the testing 
    set to the training set
    Args: 
        sentences1 (list): first list of training sentences
        sentences2 (list): second list of training sentences
        is_similar (list): list of training labels
        test_sentences1 (list): first list of testing sentences
        test_sentences2 (list): second list of testing sentences
        test_labels (list): list of testing labels
    
    Returns:
        sentences1 (list): extended list of training sentences
        sentences2 (list): extended list of training sentences
        is_similar (list): extended list of training labels
        test_sentences1 (list): shortened list of testing sentences
        test_sentences2 (list): shortened list of testing sentences
        test_labels (list): shortened list of testing labels

    """

    sentences1 += test_sentences1[len(test_sentences1)/2:]
    sentences2 += test_sentences2[len(test_sentences2)/2:]
    is_similar += test_labels[len(test_labels)/2:]

    test_sentences1 = test_sentences1[:len(test_sentences1)/2]
    test_sentences2 = test_sentences2[:len(test_sentences2)/2]
    test_labels = test_labels[:len(test_labels)/2]

    return sentences1, sentences2, is_similar, test_sentences1, test_sentences2, test_labels


def plot_model_d(history):
    """ 
    Plot two training graphs showing the val_loss and val_acc
    Args:
        history: The training history

    """
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()



def one_hot(a):
    """ Create a one-hot from a predicion
    
    Args:
        a (list): a list of predictions
    """
    b = np.zeros_like(a)
    b[np.where(a == np.max(a))] = 1
    return b



def train_word2vec(documents, embedding_dim):
    """
    train word2vector over traning documents


    Args:
        documents (list): list of document
        min_count (int): min count of word in documents to consider for word vector creation
        embedding_dim (int): output wordvector size
    Returns:
        word_vectors(dict): dict containing words and their respective vectors
    """
    '''
    model = Word2Vec(documents, min_count=1, size=embedding_dim)
    word_vector = model.wv
    del model
    '''
    # Split the sentences into lists of words
    
    docs = [string.lower().split() for string in documents]
    #Vectorize all words in docs
    model = Word2Vec(docs, min_count=1, size=EMBEDDING_DIM)
    word_vector = model.wv
    
    return word_vector


def create_embedding_matrix(tokenizer, word_vectors, embedding_dim):
    """
    Create embedding matrix containing word indexes and respective vectors from word vectors
    Args:
        tokenizer (keras.preprocessing.text.Tokenizer): keras tokenizer object containing word indexes
        word_vectors (dict): dict containing word and their respective vectors
        embedding_dim (int): dimention of word vector

    Returns:
        
    """
    nb_words = len(tokenizer.word_index) + 1
    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((nb_words, embedding_dim))
    print("Embedding matrix shape: %s" % str(embedding_matrix.shape))
    for word, i in word_index.items():
        embedding_vector = word_vectors[word]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
    return embedding_matrix



####### FIRST #######
def word_embed_meta_data(documents, embedding_dim):
    """
    Load tokenizer object for given vocabs list.

    Creates a tokenizer, 
    Args:
        documents (list): list of document

    Returns:
        tokenizer (keras.preprocessing.text.Tokenizer): keras tokenizer object
        embedding_matrix (dict): dict with word_index and vector mapping
    """
    documents = [string.lower() for string in documents]
    tokenizer = Tokenizer(filters='!".#$%&()*+,/:;<=>?@[\\]^_`{|}~\t\n')
    documents = remove_punct(documents)
    tokenizer.fit_on_texts(documents)
    word_vector = train_word2vec(documents, embedding_dim)
    embedding_matrix = create_embedding_matrix(tokenizer, word_vector, embedding_dim)
    del word_vector
    gc.collect()
    return tokenizer, embedding_matrix



def create_train_dev_set(tokenizer, sentences_pair, is_similar, max_sequence_length, validation_split_ratio):
    """
    Create training and validation dataset
    Args:
        tokenizer (keras.preprocessing.text.Tokenizer): keras tokenizer object
        sentences_pair (list): list of tuple of sentences pairs
        is_similar (list): list containing labels if respective sentences in sentence1 and sentence2
                           are same or not (1 if same else 0)
        max_sequence_length (int): max sequence length of sentences to apply padding
        validation_split_ratio (float): contain ratio to split training data into validation data

    Returns:
        train_data_1 (list): list of input features for training set from sentences1
        train_data_2 (list): list of input features for training set from sentences2
        labels_train (np.array): array containing similarity score for training data
        leaks_train(np.array): array of training leaks features

        val_data_1 (list): list of input features for validation set from sentences1
        val_data_2 (list): list of input features for validation set from sentences1
        labels_val (np.array): array containing similarity score for validation data
        leaks_val (np.array): array of validation leaks features
    """
    sentences1 = [x[0] for x in sentences_pair]
    sentences2 = [x[1] for x in sentences_pair]
    train_sequences_1 = tokenizer.texts_to_sequences(sentences1)
    train_sequences_2 = tokenizer.texts_to_sequences(sentences2)
    leaks = [[len(set(x1)), len(set(x2)), len(set(x1).intersection(x2))]
             for x1, x2 in zip(train_sequences_1, train_sequences_2)]

    train_padded_data_1 = pad_sequences(train_sequences_1, maxlen=max_sequence_length)
    train_padded_data_2 = pad_sequences(train_sequences_2, maxlen=max_sequence_length)
    train_labels = np.array(is_similar)
    leaks = np.array(leaks)

    shuffle_indices = np.random.permutation(np.arange(len(train_labels)))
    train_data_1_shuffled = train_padded_data_1[shuffle_indices]
    train_data_2_shuffled = train_padded_data_2[shuffle_indices]
    train_labels_shuffled = train_labels[shuffle_indices]
    leaks_shuffled = leaks[shuffle_indices]

    dev_idx = max(1, int(len(train_labels_shuffled) * validation_split_ratio))

    del train_padded_data_1
    del train_padded_data_2
    gc.collect()

    train_data_1, val_data_1 = train_data_1_shuffled[:-dev_idx], train_data_1_shuffled[-dev_idx:]
    train_data_2, val_data_2 = train_data_2_shuffled[:-dev_idx], train_data_2_shuffled[-dev_idx:]
    labels_train, labels_val = train_labels_shuffled[:-dev_idx], train_labels_shuffled[-dev_idx:]
    leaks_train, leaks_val = leaks_shuffled[:-dev_idx], leaks_shuffled[-dev_idx:]
    
    return train_data_1, train_data_2, labels_train, leaks_train, val_data_1, val_data_2, labels_val, leaks_val


def create_test_data(tokenizer, test_sentences_pair, max_sequence_length):
    """
    Create training and validation dataset
    Args:
        tokenizer (keras.preprocessing.text.Tokenizer): keras tokenizer object
        test_sentences_pair (list): list of tuple of sentences pairs
        max_sequence_length (int): max sequence length of sentences to apply padding

    Returns:
        test_data_1 (list): list of input features for training set from sentences1
        test_data_2 (list): list of input features for training set from sentences2
    """
    test_sentences1 = [x[0] for x in test_sentences_pair]
    test_sentences2 = [x[1] for x in test_sentences_pair]

    test_sequences_1 = tokenizer.texts_to_sequences(test_sentences1)
    test_sequences_2 = tokenizer.texts_to_sequences(test_sentences2)
    leaks_test = [[len(set(x1)), len(set(x2)), len(set(x1).intersection(x2))]
                  for x1, x2 in zip(test_sequences_1, test_sequences_2)]

    leaks_test = np.array(leaks_test)
    test_data_1 = pad_sequences(test_sequences_1, maxlen=max_sequence_length)
    test_data_2 = pad_sequences(test_sequences_2, maxlen=max_sequence_length)

    return test_data_1, test_data_2, leaks_test




def remove_punct(docs):
    for x in range(0, len(docs)):
        docs[x] = docs[x].replace(',','')
    return docs