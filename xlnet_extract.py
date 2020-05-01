########################################################
## Extract XLNet hidden layer features for movie reviews
########################################################
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
import torch.nn.functional as F
from keras.preprocessing.sequence import pad_sequences
from pytorch_transformers import XLNetModel, XLNetTokenizer, XLNetForSequenceClassification

from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

from tqdm import tqdm
import os
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TOTAL = 25000
MY_PATH = "aclImdb/train"

## Parses sentences of file in XLNet readable format
def get_sentences(text_file):
    with open(text_file, 'r', encoding = 'utf-8') as file:
        sents = file.read() + " [SEP] [CLS]"
        sents = sents.strip().replace("<br />", " ")
        return sents

def get_data(mypath):
    pos_reviews = sorted(os.listdir(mypath + "/pos"),
                        key = lambda x: int(x.split("_")[0]))
    neg_reviews = sorted(os.listdir(mypath + "/neg"),
                        key = lambda x: int(x.split("_")[0]))
    pos_length = len(pos_reviews) ## 12500 for our purposes
    neg_length = len(neg_reviews)

    rev= []
    for i in tqdm(range(pos_length), desc='+', ncols=80):
        rev.append(get_sentences(mypath+ "/pos/" + pos_reviews[i]))
    for j in tqdm(range(neg_length), desc='-', ncols=80):
        rev.append(get_sentences(mypath+ "/neg/" + neg_reviews[j]))
    return rev

## Transforms raw sentences into tokens and identifiers for XLNet
def get_dataloader(myPath, max_len = 128, batch_size = 50):
    ## load data
    train_revs = get_data(myPath)

    ## tokenize inputs
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=False)
    tokenized_texts = [tokenizer.tokenize(rev) for rev in train_revs]
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = pad_sequences(input_ids, maxlen=max_len, dtype="long", truncating="post", padding="post")
    print('tokenized inputs')

    # Create a mask of 1s for each token followed by 0s for padding
    attention_masks = []
    for seq in input_ids:
      seq_mask = [float(i>0) for i in seq]
      attention_masks.append(seq_mask)
    prediction_inputs = torch.tensor(input_ids, dtype = torch.long)
    prediction_masks = torch.tensor(attention_masks)
    prediction_labels = torch.zeros([len(train_revs)], dtype = torch.long)
    prediction_labels[:len(train_revs)//2] = 1
    print("loaded tensors")

    prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
    return prediction_dataloader

## Used to convert XLNet hidden layer output to smaller feature set for logistic regression
def conv_layer(batch, out_shape=None):
    ## each example in batch has (in_shape)
    ## want to change it so that each example has (out_shape)
    N = batch.shape[0]
    all_max = np.max(batch)
    result = np.zeros((N, 8,8))
    for i in range(N):
        for j in range(8):
            for k in range(8):
                mask = np.zeros((128,768))
                w, x, y, z = j*16, (j+1)*16, k*96, (k+1)*96
                mask[w:x][y:z] = 1
                result[i][j][k] = np.sum(mask*batch[i])/all_max

    return result.flatten()

## Used as a mirror from XLNet summarize methods in paper
class Summarize(nn.Module):
    def __init__(self):
        super(Summarize, self).__init__()
        # use another projection as in BERT
        self.dense = nn.Linear(768,768)
        torch.nn.init.normal_(self.dense.weight, std = 0.02)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        # x = torch.mean(x)
        x = self.dropout(x)
        # x = torch.tanh(self.dense(x))
        return x

def final_hidden(prediction_dataloader, model, device):
    dat = []
    for batch in tqdm(prediction_dataloader, desc = "forward", ncols = 80):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels= batch
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            logits = conv_layer(logits)
        dat.append(logits)
        # dat.append(np.mean(logits, axis = -1))
        label_ids = b_labels.to('cpu').numpy()

    return np.concatenate(dat)


# prediction_dataloader = torch.load("dataloader_test.pt")
train_dataloader = get_dataloader('aclImdb/train')
test_dataloader = get_dataloader("aclImdb/test")
# torch.save(prediction_dataloader, 'dataloader_test.pt')
print("got dataloader")

model = XLNetModel.from_pretrained("xlnet-base-cased")
summarize = Summarize()
model.cuda()
summarize.cuda()
print('got model')

X_train = final_hidden(train_dataloader, model, device)
np.save("xlnet_multilayer_train.npy", X_train)

X_test = final_hidden(test_dataloader, model, device)
np.save("xlnet_multilayer_test.npy", X_test)
# X = np.load("xlnetsumm_out.npy")
# X = np.squeeze(X)#.reshape(X.shape[0], X.shape[1]*X.shape[2])
# print(X.shape)
# y = np.ones(X.shape[0])
# y[X.shape[0]//2:] = 0
# y_train, y_test = y, y
#
# clf = LogisticRegression(max_iter = 200)#, penalty = 'l1', solver = 'liblinear')
# clf.fit(X_train, y_train)
# print("Accuracy: " + str(accuracy_score(y_test, clf.predict(X_test))))
