###########################################
## Main file for IMDB movie classification
###########################################
import numpy as np
import os
import re
from tqdm import tqdm as tqdm
from sklearn.linear_model import LogisticRegression as logReg
from sklearn.metrics import accuracy_score

from vectorizer import vectorize, get_word_bag

BAG_SIZE = 9000

dtype = np.float32
word_bag = get_word_bag()[:BAG_SIZE]
mypath = "aclImdb/train"

## Parse data given main folder path
def get_data(mypath, out_path, overwrite = False):
    pos_reviews = sorted(os.listdir(mypath + "/pos"),
                        key = lambda x: int(x.split("_")[0]))
    neg_reviews = sorted(os.listdir(mypath + "/neg"),
                        key = lambda x: int(x.split("_")[0]))
    pos_length = len(pos_reviews) ## 12500 for our purposes
    neg_length = len(neg_reviews)

    #check if npy file already exists, otherwise gather data
    if(os.path.exists(out_path) and not overwrite):
        x = np.load(out_path)
        for row in x:
            assert np.nan not in row
            assert np.inf not in row
        assert(x.shape == (25000, BAG_SIZE))
    else:
        x = np.zeros((pos_length + neg_length, BAG_SIZE), dtype=dtype)
        for i in tqdm(range(pos_length), desc='+', ncols=80):
            x[i] = vectorize(mypath+ "/pos/" + pos_reviews[i], word_bag)
        for j in tqdm(range(neg_length), desc='-', ncols=80):
            x[len(pos_reviews) + j] = vectorize(mypath+ "/neg/" + neg_reviews[j], word_bag)

        np.save(out_path, np.nan_to_num(x))

    y = np.zeros(pos_length + neg_length, dtype=dtype)
    y[:pos_length] = 1

    return x, y

## Generate n-gram models
def n_gram(x, x_test, input = 'content', ngram_max = 2, max_df = 0.85, max_features = None):
    from sklearn.feature_extraction.text import CountVectorizer
    ngram_vectorizer = CountVectorizer(input = input, max_df = max_df,
                        ngram_range=(1, ngram_max), max_features = max_features)
    ngram_vectorizer.fit(x)
    X = ngram_vectorizer.transform(x)
    X_test = ngram_vectorizer.transform(x_test)

    return X, X_test


## Test logistic regression model
def test_model(x, y, x_test, y_test):
    for c in [0.01, 0.05, 0.25, 0.5, 1]:
        lr = logReg(C=c)
        lr.fit(x, y)
        print ("Accuracy for C=%s, train = %s, test = %s"
               % (c, accuracy_score(y, lr.predict(x)),
                    accuracy_score(y_test, lr.predict(x_test))))

# x, y = get_data(mypath, "x_train.npy", overwrite = True)
# x_test, y_test = get_data("aclImdb/test", "x_test.npy", overwrite = True)

# test_model(x, y, x_test, y_test)

pos_train = sorted(os.listdir(mypath + "/pos"),
                    key = lambda x: int(x.split("_")[0]))
pos_train = list(map(lambda x: mypath + "/pos/" + x, pos_train))
neg_train = sorted(os.listdir(mypath + "/neg"),
                    key = lambda x: int(x.split("_")[0]))
neg_train = list(map(lambda x: mypath + "/neg/" + x, neg_train))
y = np.zeros(len(pos_train) + len(neg_train), dtype=dtype)
y[:len(pos_train)] = 1

pos_test = sorted(os.listdir("aclImdb/test" + "/pos"),
                    key = lambda x: int(x.split("_")[0]))
pos_test = list(map(lambda x: "aclImdb/test/pos/" + x, pos_test))
neg_test = sorted(os.listdir("aclImdb/test" + "/neg"),
                    key = lambda x: int(x.split("_")[0]))
neg_test = list(map(lambda x: "aclImdb/test/neg/" + x, neg_test))
y_test = np.zeros(len(pos_test) + len(neg_test), dtype=dtype)
y_test[:len(pos_test)] = 1

def filter_chars(s): # removes all non alphabet characters
    s = re.sub('<br', '', s)
    return re.sub('[^a-zA-Z]+', '', s).lower()

train = []
for text_file in (pos_train + neg_train):
    with open(text_file, 'r', encoding='utf-8') as file:
        review = ' '.join(list(map(filter_chars, file.read().split(' '))))
    train.append(review)

test = []
for text_file in (pos_test + neg_test):
    with open(text_file, 'r', encoding='utf-8') as file:
        review = ' '.join(list(map(filter_chars, file.read().split(' '))))
    test.append(review)


print("got files + y")
for i in range(1000, 9001, 1000):
    X, X_test = n_gram(train, test, input = 'content',max_features = i)
    print("max features = {0}".format(i))
    test_model(X,y,X_test,y_test)
