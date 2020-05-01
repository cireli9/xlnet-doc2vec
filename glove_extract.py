import numpy as np
from tqdm import tqdm
import os
import re

## Get stopwords
with open('base_model/stopwords.txt', 'r', encoding='utf-8') as file:
    STOP = file.readlines()

def load_glove():
    embeddings = {}
    with open("glove.6B.50d.txt", 'r', encoding = 'utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings[word] = vector
    return embeddings

## Parses sentences of file in Doc2Vec readable format
def get_clean(text_file):
    with open(text_file, 'r', encoding='utf-8') as file:
        review = list(map(lambda s: re.sub('\W', '', s).lower(),
                        file.read().split(' ')))
        review = [w for w in review if w not in STOP]
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

    return rev

def avg_review(rev, glove_len = 50, max_len = 100):
    '''
    input:
    Tokenized review

    output:
    4 average vectors by window
    '''
    step = rev.shape[0]//4
    result = []
    for i in range(4):
        part = rev[step*i:step*(i+1),:]
        result.append(np.mean(rev[step*i:step*(i+1),:], axis = 0))

    return np.concatenate(result)

def main():
    words = load_glove() ##dictionary of embeddings
    train = get_data("aclImdb/train")
    test = get_data("aclImdb/test")

    print('encoding train data')
    train_data = []
    for rev in train:
        embedded = []
        for w in rev:
            if w in words:
                embedded.append(words[w])
            else:
                embedded.append(np.zeros(50))
        embedded = avg_review(np.stack(embedded))
        train_data.append(embedded)

    np.save('glove_train.npy', np.stack(train_data))

    print('encoding test data')
    test_data = []
    for rev in test:
        embedded = []
        for w in rev:
            if w in words:
                embedded.append(words[w])
            else:
                embedded.append(np.zeros(50))
        embedded = avg_review(np.stack(embedded))
        test_data.append(embedded)

    np.save('glove_test.npy', np.stack(test_data))

main()
