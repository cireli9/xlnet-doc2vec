import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import time
import math

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from sklearn.linear_model import LogisticRegression

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW = 5

def summarize(rev, glove_len=50, w_size=3, stride = 2, max_len = 100):
    '''
    input:
    Takes beginning and end of a document for up to 512 tokens
    with GloVe embedding of dimension 50 for each token.

    output:
    gets sliding window of local context for each token by
    averaging with nearby vector embeddings
    '''
    summary = np.zeros((np.ceil(max_len/stride),glove_len))
    for i in range(np.ceil(max_len/stride)):
        window = range(i-w_size, i+w_size+1)
        average = []
        for j in window:
            if j < max_len and j >= 0:
                average.append(rev[j])
        summary[i] = np.mean(average)

    return summary.reshape(np.ceil(max_len/stride)*glove_len)

class Classifier(nn.Module):
    def __init__(self, in_size, out_size, n_classes = 2):
        super(Classifier, self).__init__()
        # linear layer
        self.dense1 = nn.Linear(in_size,out_size)
        torch.nn.init.normal_(self.dense1.weight, std = 0.02)
        self.dropout = nn.Dropout(p=0.1)
        self.dense2 = nn.Linear(out_size, n_classes)
        torch.nn.init.normal_(self.dense2.weight, std = 0.02)
        self.tan = nn.Tanh()

    def forward(self, x):
        x = self.dense1(x)
        x = self.tan(self.dropout(x))
        x = self.tan(self.dense2(x))
        x = torch.sigmoid(x)
        return x


def main_nn(epochs = 200):
    dbow_train = np.load('dbow_train.npy')
    dbow_test = np.load('dbow_test.npy')

    d2v_train = np.load('doc2vec_train.npy')
    d2v_test = np.load('doc2vec_test.npy')

    train_embeds = np.load('glove_train.npy')
    test_embeds = np.load('glove_test.npy')

    X_train = np.concatenate((dbow_train, train_embeds), axis = 1)
    X_test = np.concatenate((dbow_test, test_embeds), axis = 1)
    X_train2 = d2v_train
    X_test2 = d2v_test
    targets = np.zeros(25000)
    targets[:12500] = 1

    model = Classifier(X_train.shape[1], 400).double()
    model2 = Classifier(X_train2.shape[1], 400).double()
    model.cuda()
    model2.cuda()
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.1
    momentum = 0.9
    batch_size = 64
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum= momentum)

    loss_full = []
    acc_full = []
    print("begin training")

    ## get loss & acc for dbow+glove
    for i in range(epochs):
        optimizer.zero_grad()
        y_pred = model(Variable(torch.from_numpy(X_train).double()).cuda())
        loss = criterion(y_pred.cuda(), Variable(torch.from_numpy(targets).long()).cuda())
        loss.backward()
        optimizer.step()

        loss_full.append(loss)
        preds = model(Variable(torch.from_numpy(X_test).double()).cuda())
        acc = 1-np.sum(np.abs(np.argmax(np.array(preds.data.cpu()), axis = 1)-targets))/25000
        acc_full.append(acc)
        if i % 50 == 0:
            print("Loss at epoch {0}: {1}".format(i, loss))
            print("Accuracy at epoch {0}: {1}".format(i, acc))

    print("Max accuracy = {0}".format(max(acc_full)))

    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.1
    momentum = 0.9
    batch_size = 64
    optimizer = torch.optim.SGD(model2.parameters(), lr=learning_rate, momentum= momentum)
    loss2 = []
    acc2 = []
    ## get loss & acc for d2v
    for i in range(epochs):
        optimizer.zero_grad()
        y_pred = model2(Variable(torch.from_numpy(X_train2).double()).cuda())
        loss = criterion(y_pred.cuda(), Variable(torch.from_numpy(targets).long()).cuda())
        loss.backward()
        optimizer.step()

        loss2.append(loss)
        preds = model2(Variable(torch.from_numpy(X_test2).double()).cuda())
        acc = 1-np.sum(np.abs(np.argmax(np.array(preds.data.cpu()), axis = 1)-targets))/25000
        acc2.append(acc)
        if i % 50 == 0:
            print("Loss at epoch {0}: {1}".format(i, loss))
            print("Accuracy at epoch {0}: {1}".format(i, acc))

    print("Max accuracy = {0}".format(max(acc2)))
    return loss_full, acc_full, loss2, acc2

def logreg():
    d2v_train = np.load('doc2vec_train.npy')
    d2v_test = np.load('doc2vec_test.npy')
    print(d2v_train.shape)
    # train_embeds = np.load('glove_train.npy')
    # test_embeds = np.load('glove_test.npy')

    # X_train = np.concatenate((d2v_train, train_embeds), axis = 1)
    X_train = d2v_train
    X_test = d2v_test
    # X_test = np.concatenate((d2v_test, test_embeds), axis = 1)
    targets = np.zeros(25000)
    targets[:12500] = 1

    model = LogisticRegression(max_iter = 500, C=0.05, solver='liblinear',penalty='l1')
    model.fit(X_train, targets)
    preds = model.predict(X_test)
    acc = 1-np.sum(np.abs(preds - targets))/25000

    print("Accuracy = {0}".format(acc))

from sklearn import svm
def main_svm():
    d2v_train = np.load('doc2vec_train.npy')
    d2v_test = np.load('doc2vec_test.npy')
    print(d2v_train.shape)
    train_embeds = np.load('glove_train.npy')
    test_embeds = np.load('glove_test.npy')

    X_train = np.concatenate((d2v_train, train_embeds), axis = 1)
    X_test = np.concatenate((d2v_test, test_embeds), axis = 1)
    X_train = d2v_train
    X_test = d2v_test
    targets = np.zeros(25000)
    targets[:12500] = 1

    for c in [0.1, 0.5, 1.0, 5.0]:
        model = svm.SVC(C=c)
        model.fit(X_train, targets)
        preds = model.predict(X_test)
        acc = 1-np.sum(np.abs(preds - targets))/25000

        print("Accuracy = {0} for C = {1}".format(acc, c))


# logreg()
# main_svm()
loss_full, acc_full, loss2, acc2 = main_nn()
import matplotlib.pyplot as plt

## Bag-size vs error
epochs = list(range(1, 201))
plt.plot(epochs, loss_full, 'r')
plt.plot(epochs, loss2, 'b')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss vs Epochs")
plt.legend(("PV-DBOW + GloVe", "Concatenated D2V"))
plt.savefig('nn_loss.png', bbox_inches='tight')
plt.close()

plt.plot(epochs, acc_full, 'r')
plt.plot(epochs, acc2, 'b')
plt.xlabel("Epochs")
plt.ylabel("Test Accuracy")
plt.title("Accuracy over Epochs")
plt.legend(("PV-DBOW + GloVe", "Concatenated D2V"))
plt.savefig('nn_acc.png', bbox_inches='tight')
plt.close()
