###########################################
## Pre-processing functions for text in review
###########################################
import re
import numpy as np
from nltk.stem import WordNetLemmatizer

BAG_SIZE = 5000

# removes all non alphabet characters
def filter_chars(s):
    s = re.sub('<br', '', s)
    return re.sub('[^a-zA-Z]+', '', s).lower()

# Get the bucket of words, number of occurrences
def get_word_bag():
    bag_file = 'bag.txt'
    with open(bag_file, 'r') as bag:
        elems = list(bag.read().split('\n'))
    elems.pop() ## remove last '\n'
    for i in range(len(elems)):
        elems[i] = elems[i].split(' ')
        elems[i][1] = float(elems[i][1])
    return elems

# Returns lemmatized text of a review (without POS tagging)
def get_lemmatized_text(review):
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in review.split()])

## Converts review text file to vector of size 5000 with values set to word counts
## Order of words is same as order in bag.txt
def vectorize(text_file, word_bag, lemma = False):
    # Get bag of words and intialize word counter
    word_vec = np.zeros(len(word_bag), dtype=np.float32)
    bag_words = list(map(lambda x: x[0], word_bag))
    word_counter = dict()
    for word in word_bag:
        word_counter[word[0]] = 0

    # Get character-filtered file text
    if lemma:
        with open(text_file, 'r', encoding='utf-8') as file:
            review = list(map(filter_chars, get_lemmatized_text(file.read()).split(' ')))
    else:
        with open(text_file, 'r', encoding='utf-8') as file:
            review = list(map(filter_chars, file.read().split(' ')))

    # Count occurrences of each word in review
    for i in range(len(review)):
        if review[i] in bag_words:
            word_counter[review[i]] += 1

    # Create word vector
    for i in range(len(word_bag)):
        word_vec[i] = word_counter[word_bag[i][0]]

    return word_vec
