from model import SiameseBiLSTM
from inputHandler import word_embed_meta_data, create_test_data
from config import siamese_config
import matplotlib.pyplot as plt
from load_sick import *
from operator import itemgetter
from keras.models import load_model
import pandas as pd
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
from keras.utils.np_utils import to_categorical
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


filters = '!".#$%&()*+,/:;<=>?@[\\]^_`{|}~\t\n'  # Allow '-' for hypenated words


#########################################
###### LSTM Siamese Text Similarity #####
#########################################
def get_session(gpu_fraction=0.333):
	''' Prevents memory errors with TensorFlow
	'''
	gpu_options = \
		tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
					  allow_growth=True)
	return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

ktf.set_session(get_session())


def remove_punct(sentences):
    """
    remove punctuation from a list of sentences


    Args:
        sentences (list): list of sentences

    Returns:
        sentences (list): list of senteces without punctuation
    """
	for sentence in sentences:
		sentence = sentence.translate(None, filters)
	return sentences


def train_entailment(extend_training=False):
	"""
    train a siamese neural network on the SICK dataset using entailment


    Args:
        documents (bool): True to split the training data 75:25. False to split 50:50 
    """

	sentences1, sentences2, is_similar = load_sick_entailment(mode=TRAIN)
	test_sentences1, test_sentences2, test_labels = load_sick_entailment(mode=TEST)

	if extend_training == True:
		sentences1, sentences2, is_similar, test_sentences1, test_sentences2, test_labels = extend_train_set(sentences1, sentences2, is_similar, test_sentences1, test_sentences2, test_labels)




	sentences1 = remove_punct(sentences1)
	sentences2 = remove_punct(sentences2)
	test_sentences1 = remove_punct(test_sentences1)
	test_sentences2 = remove_punct(test_sentences2)

	# One hot encoding
	is_similar = to_categorical(is_similar, num_classes=3)
	test_labels = to_categorical(test_labels, num_classes=3)

	print(np.array(is_similar))

	####################################
	######## Word Embedding ############
	####################################


	# creating word embedding meta data for word embedding 
	tokenizer, embedding_matrix = word_embed_meta_data(sentences1 + sentences2 + test_sentences1 + test_sentences2,  siamese_config['EMBEDDING_DIM'])

	embedding_meta_data = {
		'tokenizer': tokenizer,
		'embedding_matrix': embedding_matrix
	}

	## creating sentence pairs
	sentences_pair = [(x1, x2) for x1, x2 in zip(sentences1, sentences2)]
	del sentences1
	del sentences2

	test_sentence_pairs = [(x1, x2) for x1, x2 in zip(test_sentences1, test_sentences2)]

	##########################
	######## Training ########
	##########################

	class Configuration(object):
		"""Dump stuff here"""

	CONFIG = Configuration()
	CONFIG.embedding_dim = siamese_config['EMBEDDING_DIM']
	CONFIG.max_sequence_length = siamese_config['MAX_SEQUENCE_LENGTH']
	CONFIG.number_lstm_units = siamese_config['NUMBER_LSTM']
	CONFIG.rate_drop_lstm = siamese_config['RATE_DROP_LSTM']
	CONFIG.number_dense_units = siamese_config['NUMBER_DENSE_UNITS']
	CONFIG.activation_function = siamese_config['ACTIVATION_FUNCTION']
	CONFIG.rate_drop_dense = siamese_config['RATE_DROP_DENSE']
	CONFIG.validation_split_ratio = siamese_config['VALIDATION_SPLIT']

	siamese = SiameseBiLSTM(CONFIG.embedding_dim , CONFIG.max_sequence_length, CONFIG.number_lstm_units , CONFIG.number_dense_units, 
							CONFIG.rate_drop_lstm, CONFIG.rate_drop_dense, CONFIG.activation_function, CONFIG.validation_split_ratio)

	

	best_model_path, history = siamese.train_entailment(sentences_pair, is_similar, embedding_meta_data, model_save_directory='./')

	########################
	###### Testing #########
	########################

	model = load_model(best_model_path)

	print(best_model_path)
	test_data_x1, test_data_x2, leaks_test = create_test_data(tokenizer, test_sentence_pairs,  siamese_config['MAX_SEQUENCE_LENGTH'])
	

	for x in range(0, len(test_data_x2)):
		test_1 = test_data_x1[x].reshape(1,20)
		test_2 = test_data_x2[x].reshape(1,20)
		test_l = leaks_test[x].reshape(1,3)

	plot_model_d(history,)






def train(mode='relatedness'):
	"""
	NOT YET IMPLEMENTED

	"""
	########################################
	############ Data Preperation ##########
	########################################
	sentences1, sentences2, is_similar = load_sick_entailment(mode=TRAIN)
	test_sentences1, test_sentences2, test_labels = load_sick_entailment(mode=TEST)

	sentences1 = remove_punct(sentences1)
	sentences2 = remove_punct(sentences2)
	test_sentences1 = remove_punct(test_sentences1)
	test_sentences2 = remove_punct(test_sentences2)

	labels_rate = []
	is_similar_rate = []

	for x in range(0, len(is_similar)):
		is_similar_rate.append((float(is_similar[x]) - MIN) / (MAX - MIN))
	for y in range(0, len(test_labels)):
		labels_rate.append((float(test_labels[y]) - MIN) / (MAX - MIN))


	is_similar = is_similar_rate
	test_labels = labels_rate

	####Test Data ####
	test_sentence_pairs = []

	####################################
	######## Word Embedding ############
	####################################
	# creating word embedding meta data for word embedding 
	tokenizer, embedding_matrix = word_embed_meta_data(sentences1 + sentences2,  siamese_config['EMBEDDING_DIM'])

	embedding_meta_data = {
		'tokenizer': tokenizer,
		'embedding_matrix': embedding_matrix
	}

	## creating sentence pairs
	sentences_pair = [(x1, x2) for x1, x2 in zip(sentences1, sentences2)]
	del sentences1
	del sentences2
	test_sentence_pairs = [(x1, x2) for x1, x2 in zip(test_sentences1, test_sentences2)]

	##########################
	######## Training ########
	##########################



	class Configuration(object):
		"""Dump stuff here"""

	CONFIG = Configuration()
	CONFIG.embedding_dim = siamese_config['EMBEDDING_DIM']
	CONFIG.max_sequence_length = siamese_config['MAX_SEQUENCE_LENGTH']
	CONFIG.number_lstm_units = siamese_config['NUMBER_LSTM']
	CONFIG.rate_drop_lstm = siamese_config['RATE_DROP_LSTM']
	CONFIG.number_dense_units = siamese_config['NUMBER_DENSE_UNITS']
	CONFIG.activation_function = siamese_config['ACTIVATION_FUNCTION']
	CONFIG.rate_drop_dense = siamese_config['RATE_DROP_DENSE']
	CONFIG.validation_split_ratio = siamese_config['VALIDATION_SPLIT']

	siamese = SiameseBiLSTM(CONFIG.embedding_dim , CONFIG.max_sequence_length, CONFIG.number_lstm_units , CONFIG.number_dense_units, 
							CONFIG.rate_drop_lstm, CONFIG.rate_drop_dense, CONFIG.activation_function, CONFIG.validation_split_ratio)
	
	best_model_path = siamese.train_model(sentences_pair, is_similar, embedding_meta_data, model_save_directory='./')


	########################
	###### Testing #########
	########################

	model = load_model(best_model_path)

	print(best_model_path)

	test_data_x1, test_data_x2, leaks_test = create_test_data(tokenizer, test_sentence_pairs, siamese_config['MAX_SEQUENCE_LENGTH'])
	print(model.evaluate([test_data_x1, test_data_x2, leaks_test], test_labels))


	#test_sentence_pairs = [(test_sentences1[0],test_sentences2[0]),
	#				   (test_sentences1[1],test_sentences2[1])]

	#preds = list(model.predict([test_data_x1, test_data_x2, leaks_test], verbose=1).ravel())
	#results = [(x, y, z) for (x, y), z in zip(test_sentence_pairs, preds)]
	#results.sort(key=itemgetter(2), reverse=True)
	#print results


if __name__ == '__main__':
	train_entailment(extend_training = True)