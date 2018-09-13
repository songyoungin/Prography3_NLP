import torch
import torch.nn as nn
import torch.optim as optim
from model import CNNTextClassifier
from dataloader import get_MR
import torchtext.data as data
import time, pickle
from gensim.models.keyedvectors import KeyedVectors
import numpy as np

from vis_tool import Visualizer

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_one_hot(word, word2idx, vocab_size):
    x = np.zeros(vocab_size, dtype=np.long)
    x[word2idx[word]] = 1
    return x

class Trainer(object):
    def __init__(self, config):
        self.config = config

        self.lr = config.lr
        self.num_epochs = config.num_epochs

        self.log_interval = config.log_interval
        self.test_interval = config.test_interval
        self.save_dir = config.save_dir

        self.mode = config.mode
        if len(self.config.kernel_sizes) == 1:
            self.channels = "single"
        else:
            self.channels = "multi"

        self.build_dataloaders()
        self.build_net()

    def build_dataloaders(self):
        print("Loading data...")
        self.text_field = data.Field(lower=True)
        self.label_field = data.Field(sequential=False)
        train_iter, dev_iter = get_MR(self.text_field, self.label_field,
                                      device=-1, repeat=False,
                                      batch_size=10)
        self.train_iter = train_iter
        self.dev_iter = dev_iter
        print("Prepare data complete!")

        self.vocab = self.text_field.vocab
        self.classes = {0: 'positive', 1: 'negative'}
        self.word2idx = self.text_field.vocab.stoi
        self.word2idx.pop("<unk>")
        self.word2idx.pop("<pad>")
        self.idx2word = self.text_field.vocab.itos

        # print("Loading Word2Vec...")
        #
        # word_vectors = KeyedVectors.load_word2vec_format(
        #     r"D:\Deep_learning\Data\GoogleNews-vectors-negative300.bin.gz", binary=True)
        #
        # w2v_matrix = []
        # for i in range(len(self.vocab)):
        #     word = self.idx2word[i]
        #     if word in word_vectors.vocab:
        #         w2v_matrix.append(word_vectors.word_vec(word))
        #     else:
        #         w2v_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))
        #
        # print("Prepare data complete!!!")
        # print(len(self.vocab))
        # print(len(self.classes))
        #
        # self.w2v_matrix = np.asarray(w2v_matrix)
        # with open("w2v_matrix.pkl", "wb") as f:
        #     pickle.dump(self.w2v_matrix, f)

        with open("w2v_matrix.pkl", "rb") as f:
            self.w2v_matrix = pickle.load(f)

        # print(w2v_matrix.shape)

    def build_net(self):
        model = CNNTextClassifier(len(self.vocab), len(self.classes))

        if self.config.training_model is not None:
            model.load_state_dict(torch.load(self.config.training_model))

        self.model = model.to(device)

        if self.mode == "CNN-rand":
            pass
        elif self.mode == "CNN-static":
            self.model.embedding.weight.data.copy_(torch.from_numpy(self.w2v_matrix))
            self.model.embedding.weight.requires_grad = False
        elif self.mode == "CNN-non-static":
            self.model.embedding.weight.data.copy_(torch.from_numpy(self.w2v_matrix))

        print("Prepare model complete!")

        self.vis = Visualizer()

    def train(self):
        criterion = nn.CrossEntropyLoss()

        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = optim.Adam(parameters, lr=self.lr)

        steps = 0
        best_acc = 0

        print("Learning started!")
        start_time = time.time()
        for epoch in range(self.num_epochs):
            for step, batch in enumerate(self.train_iter):
                self.model.train()
                feature, target = batch.text, batch.label
                feature = feature.permute(1, 0)  # batch first
                target.data = target.data.sub(1)  # index align

                step_batch = feature.size(0)

                feature = feature.to(device)
                target = target.to(device)

                logits = self.model(feature)
                loss = criterion(logits, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                steps += 1

                if steps % self.log_interval == 0:
                    predicted = torch.max(logits, 1)[1].view(target.size())
                    corrects = (predicted.data == target.data).sum()
                    accuracy = 100.0 * (float(corrects) / step_batch)

                    end_time = time.time()
                    print("[%d/%d] [%d/%d] time:%.3f loss:%.3f accuracy:%.3f"
                          % (epoch + 1, self.num_epochs, step + 1, len(self.train_iter),
                             end_time - start_time, loss.item(), accuracy))

                    self.vis.plot("%s-%s training loss plot" % (self.mode, self.channels), loss.item())
                    self.vis.plot("%s-%s training accuracy plot" % (self.mode, self.channels), accuracy)

                if steps % self.test_interval == 0:
                    acc = self.eval()
                    if acc > best_acc:
                        best_acc = acc
                        torch.save(self.model, "%s/%s-%s_best_acc.pth"
                                   % (self.save_dir, self.mode, self.channels))
        print("Learning finished!")
        torch.save(self.model, "%s/%s-%s_final.pth" % (self.save_dir, self.mode, self.channels))

    def eval(self):
        self.model.eval()

        avg_acc = 0.
        avg_loss = 0.

        criterion = nn.CrossEntropyLoss()

        for batch in self.dev_iter:
            feature, target = batch.text, batch.label
            feature = feature.permute(1, 0)  # batch first
            target.data = target.data.sub(1)  # index align

            step_batch = feature.size(0)

            feature = feature.to(device)
            target = target.to(device)

            logits = self.model(feature)
            loss = criterion(logits, target)

            predicted = torch.max(logits, 1)[1].view(target.size())
            corrects = (predicted.data == target.data).sum()
            accuracy = 100.0 * (float(corrects) / step_batch)

            avg_loss += loss.item() / len(self.dev_iter)
            avg_acc += accuracy / len(self.dev_iter)

        print("Evaluation- loss: %.3f accuracy: %.3f"
              % (avg_loss, avg_acc))
        self.vis.plot("%s-%s validate loss plot" % (self.mode, self.channels), avg_loss)
        self.vis.plot("%s-%s validate accuracy plot" % (self.mode, self.channels), avg_acc)

        return avg_acc