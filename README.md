# xlnet-doc2vec
IMDB movie review sentiment analysis w/ baseline models and xlnet + doc2vec combination features.

bert_extract.py - used to extract BERT features
doc2vec_extract.py - used to extract Doc2Vec features
xlnet_extract.py - used to extract XLNet features
glove_extract.py - used to extract GloVe features
classifier.py - used to concatenate features and test final results
results_getter.py - used for final result testing and hyperparameter optimization

Each script has commented code that was used to test various combinations of the models.

Hyperparameters used are as follows:
* Logistic regression: regularizer parameter = 1.0, loss = l2, maximum iterations = 500
* Random forest: number of trees = 100, minimum samples per node = 10
* Neural network: number of hidden units = 600, dropout probability = 0.1

Additional hyperparameters can be found within each script and are written in as the default.
