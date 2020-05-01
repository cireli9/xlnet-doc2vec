"""Extract pre-computed feature vectors from a PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import json
import re
import os
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

STOPWORDS = set(stopwords.words('english'))

class InputExample(object):

    def __init__(self, unique_id, text_a, text_b = None):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids

# we never account for tokens B since this is not a question-answer type of problem
def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputFeature`s."""

    features = []
    for ex_index, example in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > seq_length - 2:
            tokens_a = tokens_a[0:(seq_length - 2)]

        # add [cls] and [sep]
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features

def get_as_line(text_file):
    with open(text_file, 'r', encoding = 'utf-8') as file:
        rev = file.read().replace("\n", "")
        # word_tokens = word_tokenize(rev)
        # filtered = [w for w in word_tokens if not w in STOPWORDS]
        # rev = ' '.join(filtered)
        return rev


def get_data(mypath):
    pos_reviews = sorted(os.listdir(mypath + "/pos"),
                        key = lambda x: int(x.split("_")[0]))
    neg_reviews = sorted(os.listdir(mypath + "/neg"),
                        key = lambda x: int(x.split("_")[0]))
    pos_length = len(pos_reviews) ## 12500 for our purposes
    neg_length = len(neg_reviews)

    rev= []
    unique_id = 0
    for i in tqdm(range(pos_length), desc='+', ncols=80):
        rev.append(InputExample(unique_id=unique_id, text_a=get_as_line(mypath+ "/pos/" + pos_reviews[i])))
        unique_id+=1
    for j in tqdm(range(neg_length), desc='-', ncols=80):
        rev.append(InputExample(unique_id=unique_id, text_a=get_as_line(mypath+ "/neg/" + neg_reviews[j])))
        unique_id += 1
    return rev


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    layer_indexes = [-1, -2, -3, -4]

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased", do_lower_case=False)

    examples = get_data("aclImdb/test")

    features = convert_examples_to_features(
        examples=examples, seq_length=128, tokenizer=tokenizer)

    unique_id_to_feature = {}
    for feature in features:
        unique_id_to_feature[feature.unique_id] = feature

    model = BertModel.from_pretrained("bert-base-cased")
    model.to(device)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_example_index)
    eval_sampler = SequentialSampler(eval_data)

    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=2)
    # torch.save(eval_dataloader, 'bert_dataloader.pt')
    # eval_dataloader = torch.load("bert_dataloader.pt")
    print("got dataloader")

    model.eval()
    all_data = []
    for input_ids, input_mask, example_indices in tqdm(eval_dataloader,ncols=80,desc='model'):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)

        all_encoder_layers, _ = model(input_ids, token_type_ids=None, attention_mask=input_mask)
        for b, example_index in enumerate(example_indices):
            all_layers = []
            for (j, layer_index) in enumerate(layer_indexes):
                layer_output = all_encoder_layers[int(layer_index)].detach().cpu().numpy()
                layer_output = np.array([x for x in layer_output[b].flat])
                all_layers.append(layer_output[:100])
        all_data.append(np.stack(all_layers))
    np.save("bert_features.npy", np.concatenate(all_data, axis = 0))
    print("done")



if __name__ == "__main__":
    main()
    X = np.load("bert_features.npy")
    print(X.shape)
    # X = np.reshape(X, (25000, -1))
    y = np.ones(X.shape[0])
    y[X.shape[0]//2:] = 0
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(min_samples_leaf = 100)#, penalty = 'l1', solver = 'liblinear')
    clf.fit(X_train, y_train)
    print("Train Accuracy" + str(accuracy_score(y_train, clf.predict(X_train))))
    print("Accuracy: " + str(accuracy_score(y_test, clf.predict(X_test))))
