import torch
import torch.nn as nn
import torch.optim as optim
from torchtext import data
from model import CNNTextClassifier
from dataloader import get_MR

import time

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Trainer(object):
    def __init__(self, config):
        self.config = config

        self.lr = config.lr
        self.num_epochs = config.num_epochs

        self.log_interval = config.log_interval
        self.test_interval = config.test_interval
        self.save_dir = config.save_dir

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

    def build_net(self):
        model = CNNTextClassifier(self.text_field, self.label_field)

        if self.config.training_model is not None:
            model.load_state_dict(torch.load(self.config.training_model))

        self.model = model.to(device)
        print("Prepare model complete!")

    def train(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        steps = 0
        best_acc = 0

        print("Learning started!")
        start_time = time.time()
        for epoch in range(self.num_epochs):
            for step, batch in enumerate(self.train_iter):
                self.model.train()
                feature, target = batch.text, batch.label
                feature = feature.permute(1, 0)  # batch first
                target.data = target.data.sub(1) # index align

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
                          %(epoch+1, self.num_epochs, step+1, len(self.train_iter),
                            end_time-start_time, loss.item(), accuracy))

                if steps % self.test_interval == 0:
                    acc = self.eval()
                    if acc > best_acc:
                        best_acc = acc
                        torch.save(self.model, "%s/best_acc.pth"
                                   % (self.save_dir))
        print("Learning finished!")
        torch.save(self.model, "%s/final.pth" % self.save_dir)

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

            avg_loss += loss / len(self.dev_iter)
            avg_acc += accuracy / len(self.dev_iter)

        print("Evaluation- loss: %.3f accuracy: %.3f"
              % (avg_loss, avg_acc))

        return avg_acc