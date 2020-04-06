###########################################
## Doc2Vec implmentation for IMDB classification
###########################################
import numpy as np
import os
import re
import random
import pickle
from tqdm import tqdm as tqdm
from warnings import filterwarnings

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
import gensim.models.doc2vec
from collections import namedtuple
import multiprocessing

MY_PATH = "aclImdb/train"

## Parses sentences of file in Doc2Vec readable format
def get_clean(text_file):
    with open(text_file, 'r', encoding='utf-8') as file:
        review = list(map(lambda s: re.sub('\W', '', s).lower(),
                        file.read().split(' ')))
    return review

## Parse data given main folder path
def get_data(path):
    pos_reviews = sorted(os.listdir(path + "/pos"),
                        key = lambda x: int(x.split("_")[0]))
    neg_reviews = sorted(os.listdir(path + "/neg"),
                        key = lambda x: int(x.split("_")[0]))
    pos_length = len(pos_reviews) ## 12500 for our purposes
    neg_length = len(neg_reviews)

    rev= []
    for i in tqdm(range(pos_length), desc='+', ncols=80):
        rev.append(get_clean(path+ "/pos/" + pos_reviews[i]))
    for j in tqdm(range(neg_length), desc='-', ncols=80):
        rev.append(get_clean(path+ "/neg/" + neg_reviews[j]))

    docs = []
    SentimentDoc = namedtuple('SentimentDoc', 'words tags sentiment')
    for i, text in enumerate(rev):
        if i < len(rev)//2:
            docs.append(SentimentDoc(text, [i], 1))
        else:
            docs.append(SentimentDoc(text, [i], 0))
    return docs

doc_list = get_data(MY_PATH)
test_list = get_data("aclImdb/test")

## Get models
# # PV-DM w/average
# Doc2Vec(dm=1, dm_mean=1, size=100, window=10, negative=5, hs=0, min_count=2, workers=cores),
cores = multiprocessing.cpu_count()
assert gensim.models.doc2vec.FAST_VERSION > -1
simple_models = [
    # PV-DM w/concatenation - window=5 (both sides) approximates paper's 10-word total window size
    Doc2Vec(dm=1, dm_concat=1, size=100, window=5, negative=5, hs=0, min_count=2, workers=cores),
    # PV-DBOW
    Doc2Vec(dm=0, size=100, negative=5, hs=0, min_count=2, workers=cores)
]

# speed setup by sharing results of 1st model's vocabulary scan
simple_models[0].build_vocab(doc_list)  # PV-DM/concat requires one special NULL word so it serves as template
for model in simple_models[1:]:
    model.reset_from(simple_models[0])
print("got model")

#Create dbow+dmm model (concatenation of model 2 and 3)
train_model = simple_models[1]

#Load XLNet features
xlnet_train = np.load("xlnet_sequence_out.npy").tolist()
xlnet_features = np.load("xlnet_seq_test_out.npy").tolist()
alpha, min_alpha, passes = (0.025, 0.001, 20)
alpha_delta = (alpha - min_alpha) / passes
print("begin evaluation")
accs = []
filterwarnings('ignore')
for epoch in range(passes):
    ## Shuffle data
    # random.seed(epoch)
    docs_features = random.sample(list(zip(doc_list, xlnet_train)), len(doc_list))
    random.seed(epoch)
    doc_list = random.sample(doc_list, len(doc_list))
    test_features = list(zip(test_list, xlnet_features))
    ## Train doc2vec
    train_model.alpha, train_model.min_alpha = alpha, alpha
    train_model.train(doc_list, total_examples=train_model.corpus_count, epochs = 1)

    ## Logistic regression
    clf = LogisticRegression(max_iter = 200)
    # normalize doc2vec features to match order of xlnet features
    norm_factors =[10**(-np.log10(np.max(train_model.docvecs[docs_features[i][0].tags[0]])) - 1)
                    for i in range(len(docs_features))]
    train_targets, train_regressors = zip(*[(doc[0].sentiment,
                np.concatenate((norm_factors[i]*train_model.docvecs[doc[0].tags[0]],doc[1]), axis = None))
                for i, doc in enumerate(docs_features)])
    test_targets, test_regressors = zip(*[(doc[0].sentiment,
                np.concatenate((norm_factors[i]*train_model.docvecs[doc[0].tags[0]],doc[1]), axis = None))
                for i, doc in enumerate(test_features)])
    # train_targets, train_regressors = zip(*[(doc.sentiment,
    #             10*train_model.docvecs[doc.tags[0]]) for doc in doc_list])
    # test_targets, test_regressors = zip(*[(doc.sentiment,
    #             10*train_model.docvecs[doc.tags[0]]) for doc in test_list])
    clf.fit(train_regressors, train_targets)
    err = accuracy_score(test_targets, clf.predict(test_regressors))
    accs.append(err)
    print("%f : %i passes at alpha %f" % (err, epoch + 1, alpha))
    alpha -= alpha_delta

print(accs)
