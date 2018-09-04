import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from  konlpy.tag import Kkma
import pickle
import os

torch.manual_seed(1)

# device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define Korean corpus
korTagger = Kkma()
corpus = ["나는 책을 읽는다.",
          "나는 너를 사랑한다.",
          "철수는 학교를 간다.",
          "영희는 달린다.",
          "대한민국의 수도는 서울이다.",
          "중국의 수도는 베이징이다.",
          "겨울은 춥다.",
          "여름은 덥다."]

# tokenize sentences
tokenized_corpus = [korTagger.morphs(sentence) for sentence in corpus]
print(tokenized_corpus)

# creating vocabulary
word2idx = {}
idx2word = {}

for sentence in tokenized_corpus:
    for token in sentence:
        if token not in word2idx:
            idx = len(word2idx)
            word2idx[token] = idx
            idx2word[idx] = token

print(word2idx)
print(idx2word)

word_vocab_size = len(word2idx)

# save vocabulary for reproducibility
pickle_path = "./pickles"
os.makedirs(pickle_path, exist_ok=True)
with open(pickle_path + "/word2idx.pkl", "wb") as f:
    pickle.dump(word2idx, f)
with open(pickle_path + "/idx2word.pkl", "wb") as f:
    pickle.dump(idx2word, f)

# generate center, context word pairs
idx_pairs = []
window_size = 2

for sentence in tokenized_corpus:
    word_indices = [word2idx[token] for token in sentence]
    for center_word_idx in range(len(word_indices)):
        for w in range(-window_size, window_size):
            context_word_idx = center_word_idx + w

            if context_word_idx < 0 or \
                context_word_idx >= len(word_indices) or \
                center_word_idx == context_word_idx:
                continue

            idx_pairs.append((word_indices[center_word_idx], word_indices[context_word_idx]))

idx_pairs = np.array(idx_pairs)

with open(pickle_path + "/idx_pairs.pkl", "wb") as f:
    pickle.dump(idx_pairs, f)

# define one-hot encoding function
def get_one_hot(word_idx, word_vocab_size):
    x = torch.zeros(word_vocab_size).float().to(device)
    x[word_idx] = 1.
    return x

# define embedding matrices
embedding_dim = 5
W1 = torch.randn((embedding_dim, word_vocab_size), requires_grad=True).float().to(device)
W2 = torch.randn((word_vocab_size, embedding_dim), requires_grad=True).float().to(device)

# training
num_epochs = 100
lr = 0.01

print("Learning started!!!")
for epoch in range(num_epochs):
    loss_val = 0.
    for center, context in idx_pairs:
        x = get_one_hot(center, word_vocab_size).float().to(device)
        y_true = torch.from_numpy(np.array([context])).long().to(device)

        z1 = torch.matmul(W1, x)
        z2 = torch.matmul(W2, z1)

        log_softmax = F.log_softmax(z2, dim=0).view(1, -1)
        loss = F.nll_loss(log_softmax, y_true)
        loss_val += loss.item()
        loss.backward()

        W1.data -= lr * W1.grad.data
        W2.data -= lr * W2.grad.data

        W1.grad.data.zero_()
        W2.grad.data.zero_()

    if epoch % 10 == 0:
        avg_loss = loss_val / float(len(idx_pairs))
        print("loss:", avg_loss)

print("Learning finished!!!")

# save embedding matrices
with open(pickle_path + "/W1.pkl", "wb") as f:
    pickle.dump(W1, f)
with open(pickle_path + "/W2.pkl", "wb") as f:
    pickle.dump(W2, f)