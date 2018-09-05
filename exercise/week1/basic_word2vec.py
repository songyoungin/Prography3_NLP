import torch
import torch.nn.functional as F

from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

import numpy as np
from  konlpy.tag import Kkma

import pickle
import os, shutil

torch.manual_seed(1)

# device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(pickle_path):
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

def test(pickle_path):
    with open(pickle_path + "/W1.pkl", "rb") as f:
        W1 = pickle.load(f).detach().cpu().numpy()
    with open(pickle_path + "/W2.pkl", "rb") as f:
        W2 = pickle.load(f).detach().cpu().numpy()
    with open(pickle_path + "/word2idx.pkl", "rb") as f:
        word2idx = pickle.load(f)

    W1_t = np.transpose(W1, (1, 0))

    embedding_mat = W1_t + W2 / 2.0
    print(embedding_mat)

    test_word = "수도"

    print(word2idx)

    test_idx = word2idx[test_word]
    test_vec = W1_t[test_idx]

    print(test_word)
    print(test_idx)
    print(test_vec)

def visualize(pickle_path):
    # load embedding matrices from pkl files
    with open(pickle_path + "/W1.pkl", "rb") as f:
        W1 = pickle.load(f).detach().cpu().numpy()
    with open(pickle_path + "/W2.pkl", "rb") as f:
        W2 = pickle.load(f).detach().cpu().numpy()

    # load vocabulary from pkl files
    with open(pickle_path + "/word2idx.pkl", "rb") as f:
        word2idx = pickle.load(f)

    # compute correlation matrices
    corr_W1 = np.matmul(np.transpose(W1, (1, 0)), W1)
    corr_W2 = np.matmul(W2, np.transpose(W2, (1, 0)))
    groups = list(word2idx.keys())
    groups = [key.encode('utf-8') + b'\n' for key in groups]
    print(groups)

    # visualize W1 using matplotlib
    # plt.figure(figsize=(10, 10))
    # plt.matshow(corr_W1, fignum=1)
    # x_pos = np.arange(len(groups))
    # plt.xticks(x_pos, groups)
    # y_pos = np.arange(len(groups))
    # plt.yticks(y_pos, groups)
    # plt.show()
    #
    # # visualize W2 using matplotlib
    # plt.figure(figsize=(10, 10))
    # plt.matshow(corr_W2, fignum=1)
    # x_pos = np.arange(len(groups))
    # plt.xticks(x_pos, groups)
    # y_pos = np.arange(len(groups))
    # plt.yticks(y_pos, groups)
    # plt.show()

    # visualize W1 using tensorboard
    # try:
    #     shutil.rmtree('runs/')
    # except:
    #     pass

    writer = SummaryWriter("./tensorboard")
    writer.add_embedding(np.transpose(W1, (1, 0)), metadata=groups)
    writer.close()

if __name__ == "__main__":
    pickle_path = "./pickles"

    # train(pickle_path)
    # visualize(pickle_path)
    test(pickle_path)