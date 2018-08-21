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


rte_lmappings = {'contradiction': np.array([1,0,0]), 'neutral': np.array([0,1,0]), 'entailment': np.array([0,0,1])}

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