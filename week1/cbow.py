import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pandas as pd
import pickle, os
from sklearn.utils import shuffle
from tensorboardX import SummaryWriter

torch.manual_seed(1)

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(CBOW, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim * context_size, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, x):
        emb = self.embedding(x).view(1, -1)
        out = F.relu(self.linear1(emb))
        out = self.linear2(out)
        prob = F.log_softmax(out, dim=1)
        return prob

class Worker(object):
    def __init__(self, dataroot, embedding_dim, context_size):
        # load data from csv file
        df = pd.read_csv(dataroot, engine='python')
        df = shuffle(df)
        data = df.values[:2000, 1]
        tokenized = [sentence.split() for sentence in data]

        os.makedirs("pickles", exist_ok=True)

        with open("pickles/news_tokenize.pkl", "wb") as file:
            pickle.dump(tokenized, file)

        # setup CBOW data
        cbows = []
        for sentence in tokenized:
            for i in range(context_size, len(sentence) - context_size):
                context = [sentence[i - 2], sentence[i - 1], sentence[i + 1], sentence[i + 2]]
                target = [sentence[i]]
                cbows.append((context, target))

        # setup vocabulary
        word2idx = {}
        for sentence in tokenized:
            for token in sentence:
                if token not in word2idx:
                    word2idx[token] = len(word2idx)

        self.tokenized = tokenized
        self.cbows = cbows
        self.word2idx = word2idx
        self.idx2word = {v: k for k, v in word2idx.items()}

        self.embedding_dim = embedding_dim
        self.context_size = context_size

    def main(self):
        self.train()
        self.test()
        self.visualize()

    def train(self):
        vocab_size = len(self.word2idx)

        lr = 0.05
        num_epochs = 10

        criterion = nn.NLLLoss()
        self.model = CBOW(vocab_size, self.embedding_dim, self.context_size * 2)
        optimizer = optim.SGD(self.model.parameters(), lr=lr)

        print("Learning started!!!")
        for epoch in range(num_epochs):
            for step, (context, target) in enumerate(self.cbows):
                context_idx = [self.word2idx[w] for w in context]
                context_idx = torch.LongTensor(context_idx)
                target_idx = torch.LongTensor([self.word2idx[target[0]]])

                optimizer.zero_grad()
                outputs = self.model(context_idx)
                loss = criterion(outputs, target_idx)
                loss.backward()
                optimizer.step()

                if step % 1000 == 0:
                    print("[%d/%d] [%d/%d] loss: %.3f" % (epoch + 1, num_epochs, step + 1, len(self.cbows), loss.item()))

        print("Learning finished!")
        self.model.cpu()
        torch.save(self.model, "weights/cbow_newsCorpus.pth")
        print("Saving model completed!!!")

    def test(self):
        self.model = torch.load("weights/cbow_newsCorpus.pth")

        idx2word = {v: k for k, v in self.word2idx.items()}
        test_input = self.cbows[0]

        test_input_idx = [self.word2idx[token] for token in test_input[0]]
        test_input_idx = torch.LongTensor(test_input_idx)
        test_target_idx = [self.word2idx[test_input[1][0]]]

        self.model.eval()
        outputs = self.model(test_input_idx)
        outputs, idx = torch.max(outputs, dim=1)

        print("Predicted:", idx2word[idx.data[0].item()], " Actual:", idx2word[test_target_idx[0]])

    def visualize(self):
        logdir = "tensorboard/cbow"
        os.makedirs(logdir, exist_ok=True)

        mat = self.model.embedding.weight.data
        groups = [self.idx2word[i] for i in range(len(self.word2idx))]

        writer = SummaryWriter(logdir)
        writer.add_embedding(mat, metadata=groups)
        writer.close()

if __name__ == "__main__":
    worker = Worker(dataroot="abcnews-date-text.csv",
                    embedding_dim=10, context_size=2)
    worker.main()