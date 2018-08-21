LSTM Siamese neural network for predicting sentence entailment
===============================================================================
The Sentences Involving Compositional Knowledge (SICK) dataset consists of 9,840 pairs of
sentences. Each sentence pair is labelled as either contradiction, neutral or entailment. This repo uses a deep, Siamese, bidirectional, Long Short-Term Memory (LSTM) network to predict sentence entailment using Word2Vec embeddings. The data set is split into 4,934 training pairs and 4,906 test pairs.

Usage
-------------------------------------------------------------------------------
Run:
- python controller.py


Author & Credit
-------------------------------------------------------------------------------
This repo is an adaptation of the brilliant Siamese neural network implmentation by Aman Srivastava.
- https://github.com/amansrivastava17/lstm-siamese-text-similarity
- https://github.com/amansrivastava17