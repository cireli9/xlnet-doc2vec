########################################################
## Extract XLNet hidden layer features for movie reviews
########################################################

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
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
        sents = file.read().split('.')
        exclude = []
        for i in range(len(sents)):
            if sents[i].isspace() or sents[i] == "":
                exclude.append(i)
            else:
                sents[i] = sents[i] + " [SEP]"
            if i == len(sents) - 1:
                sents[i] = sents[i] + " [CLS]"

        sents = [sents[i] for i in range(len(sents)) if i not in exclude]
        return ' '.join(sents)

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
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)
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

    return result

## Test logistic regression model for large data
def test_model(dat, batch_size = 50):
    N = len(dat)*batch_size
    lr = SGDClassifier(loss = 'log')
    for idx, batch in enumerate(dat):
        if idx < N//(batch_size*2):
            lr.partial_fit(batch, np.ones(50), classes = np.array([0,1]))
        else:
            lr.partial_fit(batch, np.zeros(50))
    print("fit model")

    correct = 0
    all_preds = []
    for idx, batch in enumerate(dat):
        preds = lr.predict(batch)
        all_preds += list(preds)
        if idx < N//(batch_size*2):
            correct += accuracy_score(np.ones(50), preds,normalize= False)
        else:
            correct+= accuracy_score(np.zeros(50), preds,normalize= False)
    print ("Accuracy for alpha=%s,  = %s" % (0.0001, correct/N))

    with open("output.txt", 'w') as file:
	    for pred in all_preds:
	        file.write(str(int(pred)) + '\n')


# prediction_dataloader = torch.load("dataloader.pt")
prediction_dataloader = get_dataloader("aclImdb/test")
torch.save(prediction_dataloader, 'dataloader_test.pt')
print("got dataloader")

model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased")
model.cuda()
print('got model')

dat = []
losses = []
for batch in tqdm(prediction_dataloader, desc = "forward", ncols = 80):
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels= batch
    with torch.no_grad():
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        logits = outputs[0]
    logits = logits.detach().cpu().numpy()
    dat.append(logits)
    label_ids = b_labels.to('cpu').numpy()

X = np.concatenate(dat)
np.save("xlnet_seq_test_out.npy", X)
# X = np.load("xlnet_sequence_out.npy")
# X = X.reshape(X.shape[0], X.shape[1]*X.shape[2])
y = np.ones(X.shape[0])
y[X.shape[0]//2:] = 0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X.shape)
clf = RandomForestClassifier(min_samples_leaf = 100)
clf.fit(X_train, y_train)
print("Accuracy: " + str(accuracy_score(y_test, clf.predict(X_test))))
