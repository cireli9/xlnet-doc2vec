###########################################
## Doc2Vec implmentation for IMDB classification
###########################################
import numpy as np
import os
import re
import random
import pickle
from tqdm import tqdm as tqdm
import time
from warnings import filterwarnings

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
import gensim.models.doc2vec
from collections import namedtuple
import multiprocessing

MY_PATH = "aclImdb/train"

# with open('base_model/stopwords.txt') as f:
#     SW = [word for line in f for word in line.split()]

## Parses sentences of file in Doc2Vec readable format
def get_clean(text_file):
    with open(text_file, 'r', encoding='utf-8') as file:
        review = list(map(lambda s: re.sub('\W', '', s).lower(),
                        file.read().split(' ')))
    return review

## Parse data given main folder path
def get_data(path, test = False):
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
            if test:
                docs.append(SentimentDoc(text, [i+25000], 1))
            else:
                docs.append(SentimentDoc(text, [i], 1))
        else:
            if test:
                docs.append(SentimentDoc(text, [i+25000], 0))
            else:
                docs.append(SentimentDoc(text, [i], 0))
    return docs

doc_list = get_data(MY_PATH)
test_list = get_data("aclImdb/test", test=True)
all_docs = doc_list + test_list

## Get models
# # PV-DM w/average
# Doc2Vec(dm=1, dm_mean=1, size=100, window=10, negative=5, hs=0, min_count=2, workers=cores),
cores = multiprocessing.cpu_count()
simple_models = [
    # PV-DM w/concatenation - window=5 (both sides) approximates paper's 10-word total window size
    Doc2Vec(dm=1, dm_concat=1, size=100, window=5, negative=5, hs=0, min_count=2, workers=cores),
    # PV-DBOW
    Doc2Vec(dm=0, size=100, negative=5, hs=0, min_count=2, workers=cores),
    # PV-DM w/average
    Doc2Vec(dm=1, dm_mean=1, size=100, window=10, negative=5, hs=0, min_count=2, workers=cores)]

# speed setup by sharing results of 1st model's vocabulary scan
simple_models[0].build_vocab(all_docs)  # PV-DM/concat requires one special NULL word so it serves as template
for model in simple_models[1:]:
    model.reset_from(simple_models[0])
print("got model")

#Create dbow+dmm model (concatenation of model 2 and 3)
# train_model = simple_models[1]
model1 = simple_models[1]
model2 = simple_models[2]

# bert_train = np.load("xlnetsumm_out.npy")
# bert_test = np.load("xlnetsumm1_out.npy")
alpha, min_alpha, passes = (0.025, 0.001, 20)
alpha_delta = (alpha - min_alpha) / passes
print("begin evaluation")
accs = []
filterwarnings('ignore')

start = time.time()
for epoch in range(passes):
    ## Shuffle data
    random.seed(epoch)
    all_docs = random.sample(all_docs, len(all_docs))

    ## Train doc2vec
    model1.alpha, model1.min_alpha, model2.alpha, model2.min_alpha = alpha,alpha,alpha,alpha
    # train_model.alpha, train_model.min_alpha = alpha, alpha
    model1.train(all_docs, total_examples = 50000, epochs = 1)
    model2.train(all_docs, total_examples = 50000, epochs = 1)
    # train_model.train(all_docs, total_examples=50000, epochs=1)
    ## Logistic regression
    clf = RandomForestClassifier(min_samples_leaf=20)
    clf2 = LogisticRegression(max_iter=200)
    # test_features = [10*np.concatenate((np.array(model1.docvecs[doc.tags[0]]),
    #                 np.array(model2.docvecs[doc.tags[0]])))
    #                 for idx, doc in enumerate(test_list)]
    # train_features = [10*np.concatenate((np.array(model1.docvecs[doc.tags[0]]),
    #                 np.array(model2.docvecs[doc.tags[0]])))
    #                 for idx, doc in enumerate(doc_list)]
    test_features = [10*np.array(train_model.docvecs[doc.tags[0]])
                    for idx, doc in enumerate(test_list)]
    train_features = [10*np.array(train_model.docvecs[doc.tags[0]])
                    for idx, doc in enumerate(doc_list)]

    # test_targets, test_regressors = zip(*[(doc.sentiment,
    #             np.concatenate((10*np.array(model1.docvecs[doc.tags[0]]),
    #             bert_test[idx,:],10*np.array(model2.docvecs[doc.tags[0]]))))
    #             for idx, doc in enumerate(test_list)])
    # train_targets, train_regressors = zip(*[(doc.sentiment,
    #             np.concatenate((10*np.array(model1.docvecs[doc.tags[0]]),
    #             bert_train[idx,:],10*np.array(model2.docvecs[doc.tags[0]]))))
    #             for idx, doc in enumerate(doc_list)])

    train_targets, train_regressors = zip(*[(doc.sentiment,
                10*np.array(train_model.docvecs[doc.tags[0]]))
                for idx, doc in enumerate(doc_list)])
    test_targets, test_regressors = zip(*[(doc.sentiment,
                10*np.array(train_model.docvecs[doc.tags[0]]))
                for idx, doc in enumerate(test_list)])

    # train_targets, train_regressors = zip(*[(doc.sentiment,
    #             10*np.concatenate((np.array(model1.docvecs[doc.tags[0]]),
    #             np.array(model2.docvecs[doc.tags[0]]))))
    #             for idx, doc in enumerate(doc_list)])
    # test_targets, test_regressors = zip(*[(doc.sentiment,
    #             10*np.concatenate((np.array(model1.docvecs[doc.tags[0]]),
    #             np.array(model2.docvecs[doc.tags[0]]))))
    #             for idx, doc in enumerate(test_list)])
    clf.fit(train_regressors, train_targets)
    clf2.fit(train_regressors, train_targets)



    err = accuracy_score(test_targets, clf.predict(test_regressors))
    err2 = accuracy_score(test_targets, clf2.predict(test_regressors))
    accs.append(err2)
    print("RF %f : %i passes at alpha %f" % (err, epoch + 1, alpha))
    print("LR %f : %i passes at alpha %f" % (err2, epoch + 1, alpha))
    alpha -= alpha_delta

print("Time is {0}".format(time.time()-start))
np.save('dbow_train.npy', np.stack(train_features))
np.save('dbow_test.npy', np.stack(test_features))
