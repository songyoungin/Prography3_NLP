import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from collections import Counter
import pandas as pd
from  tensorboardX import SummaryWriter
import os, pickle, random
from sklearn.utils import shuffle

flatten = lambda l: [item for sublist in l for item in sublist]

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class WordPairDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

# u: center word
# v: context(target) word
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()

        self.u_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.v_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim

        self.init_embeddings()

    def init_embeddings(self):
        range = 0.5 / self.embedding_dim
        self.u_embeddings.weight.data.uniform_(-range, range)
        self.v_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, pos_u, pos_v, neg_v):
        losses = []
        emb_u = self.u_embeddings(pos_u)
        # print("positive embedding u:", emb_u.shape)

        emb_v = self.v_embeddings(pos_v)
        # print("positive embedding v:", emb_v.shape)

        pos_score = torch.mul(emb_u, emb_v).squeeze()
        pos_score = torch.sum(pos_score, dim=1)
        pos_score = F.logsigmoid(pos_score)

        losses.append(sum(pos_score))

        neg_emb_v = self.v_embeddings(neg_v)
        # print("negative embedding v:", neg_emb_v.shape)

        neg_score = torch.bmm(neg_emb_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = torch.sum(neg_score, dim=1)
        neg_score = F.logsigmoid(-1 * neg_score)

        losses.append(sum(neg_score))

        return -1 * sum(losses)

def negative_sampling(word2idx, targets, unigram_table, k):
    batch_size = targets.size(0)
    neg_samples = []
    for i in range(batch_size):
        nsample = []
        target_index = targets[i].item()
        while len(nsample) < k:
            neg = random.choice(unigram_table)
            if word2idx[neg] == target_index:
                continue
            nsample.append(neg)
        neg_samples.append(torch.LongTensor([word2idx[w] for w in nsample]).view(1, -1))
    # print(neg_samples)
    return torch.cat(neg_samples)

def train(idx_pairs, word2idx, unigram_table):
    embedding_dim = 10
    num_epochs = 10
    negnum = 10
    lr = 0.05
    batch_size = 10

    # setup dataset, dataloader
    train_data = WordPairDataset(idx_pairs)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    model = Word2Vec(len(word2idx), embedding_dim).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    print("Learning started!!!")
    for epoch in range(num_epochs):
        for step, (inputs, targets) in enumerate(train_loader):
            inputs = torch.LongTensor(inputs).to(device)
            targets = torch.LongTensor(targets).to(device)
            negs = negative_sampling(word2idx, targets, unigram_table, negnum).to(device)

            loss = model(inputs, targets, negs)

            model.zero_grad()
            loss.backward()
            optimizer.step()

            # do logging
            if step % 500 == 0:
                print("[%d/%d] [%d/%d] loss:%.3f"
                      % (epoch+1, num_epochs, step, len(train_loader), loss.item()))
    print("Learning finished!!!")

    # save trained model
    model.cpu()
    torch.save(model, "weights/word2vec_newsCorpus.pth")
    print("Saving model completed!!!")

def visualize(model_path, idx2word):
    logdir = "tensorboard/word2vec"
    os.makedirs(logdir,exist_ok=True)

    model = torch.load(model_path)
    print(model)

    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    # setup matrix and metadata
    matrix = (model.u_embeddings.weight.data + model.v_embeddings.weight.data) / 2.
    label = [idx2word[i] for i in range(len(idx2word))]

    writer.add_embedding(matrix, metadata=label)
    writer.close()

def main():
    # load data from pickle file
    with open("pickles/news_tokenize.pkl", "rb") as f:
        tokenized = pickle.load(f)

    # subsampling
    word_count = Counter(flatten(tokenized))
    min_count = 2
    stopwords = []
    for w, c in word_count.items():
        if c < min_count:
            stopwords.append(w)

    # setup word2idx, idx2word
    vocab = list(set(flatten(tokenized)) - set(stopwords))
    word2idx = {'<unk>': 0}
    for word in vocab:
        if word not in word2idx:
            word2idx[word] = len(word2idx)
    idx2word = {v: k for k, v in word2idx.items()}

    # make word pairs
    window_size = 2
    idx_pairs = []
    for sentence in tokenized:
        word_idxs = [word2idx[w] for w in sentence if w in word2idx]
        for center_word_pos in range(len(word_idxs)):
            for w in range(-window_size, window_size + 1):
                context_word_pos = center_word_pos + w

                if context_word_pos < 0 or context_word_pos >= len(word_idxs) or center_word_pos == context_word_pos:
                    continue

                idx_pairs.append((word_idxs[center_word_pos], word_idxs[context_word_pos]))

    # build unigram distribution
    Z = 0.001
    num_total_words = sum([c for w, c in word_count.items() if w not in stopwords])
    unigram_table = []

    for word in vocab:
        unigram_table.extend([word] * int(((word_count[word] / num_total_words) ** 0.75) / Z))

    # save idx2word for reproducibility
    os.makedirs("pickles", exist_ok=True)
    with open("pickles/word2vec_idx2word.pkl", "wb") as f:
        pickle.dump(idx2word, f)

    train(idx_pairs, word2idx, unigram_table)
    visualize("weights/word2vec_newsCorpus.pth", idx2word)

if __name__ == "__main__":
    main()