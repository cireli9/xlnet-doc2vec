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
from sklearn.model_selection import train_test_split

filterwarnings('ignore')

X_train = np.load('dbow_train.npy').reshape((25000, -1))
# X_train = np.concatenate((X_train, np.load('dbow_train.npy')), axis=1)
X_test = np.load('dbow_test.npy').reshape((25000, -1))
# X_test = np.concatenate((X_test, np.load('dbow_test.npy')), axis=1)
print(X_train.shape)
# X_test = np.load('xlnetsumm1_out.npy').reshape((25000, -1))
targets = np.zeros(X_train.shape[0])
targets[:12500] = 1
y_train, y_test = targets, targets

# X_train, X_test, y_train, y_test = train_test_split(X_train, targets, test_size=0.3, random_state=42)

# X_test = np.load('bert_test_features.npy').reshape((25000, -1))

clf1 = LogisticRegression()
clf1.fit(X_train, y_train)
preds = clf1.predict(X_test)
acc = 1-np.sum(np.abs(preds-y_test))/25000
print("Accuracy LR: {0}".format(acc))

clf2 = RandomForestClassifier()
clf2.fit(X_train, y_train)
preds = clf2.predict(X_test)
acc = 1-np.sum(np.abs(preds-y_test))/25000
print("Accuracy RF: {0}".format(acc))
