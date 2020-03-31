#################################################
## Create bag of words given a full movie review
## including removal of stopwords and punctuation
#################################################

import re
import os
from os.path import isfile, join
from nltk.stem import WordNetLemmatizer

NUM_REVIEWS = 12500 # count of each class
TOTAL_REVIEWS = NUM_REVIEWS * 2 # positive + negative
BAG_SIZE = 9000
mypath = "aclImdb/train"
with open('stopwords.txt') as f:
    sw = [word for line in f for word in line.split()]

## Removes all non alphabet characters
def filter_chars(s):
    s = re.sub('<br', '', s)
    return re.sub('[^a-zA-Z]+', '', s).lower()

## Get lemmatized text of a review
def get_lemmatized_text(review):
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in review.split()])

bag = {}

## Create bag from a review
def bag_reviews(path, fileName):
    fullpath = path + "/{0}".format(fileName)
    f = open(fullpath, "r", encoding = "utf8")
    # text = f.read()
    # lemmatized_review = get_lemmatized_text(text)
    for word in f.read().split():
        word = filter_chars(word)
        if len(word) == 0 or word in sw:
            continue
        if word in bag:
            bag[word] = bag[word] + 1
        else:
            bag[word] = 1
    f.close()

for f in os.listdir(mypath + "/neg"):
    bag_reviews(mypath+"/neg", f)
for f in os.listdir(mypath+"/pos"):
    bag_reviews(mypath+"/pos", f)

f = open("bag_lemma.txt", "w", encoding = "utf8")
words = bag.items()
def getval(x):
    return -x[1]
sortedwords = sorted(words, key = getval)
for word in sortedwords[0:min(BAG_SIZE, len(sortedwords))]:
    #  word is a tuple (word, occurences)
    f.write("{0} {1:.3f}\n".format(word[0], word[1]))
f.close()
print("created word dictionary")
