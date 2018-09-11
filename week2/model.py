import torch
import torch.nn as nn
import torch.nn.functional as F

from config import get_config

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CNNTextClassifier(nn.Module):
    def __init__(self, text_field, label_field):
        super(CNNTextClassifier, self).__init__()
        config = get_config()
        config.vocab_size = len(text_field.vocab)
        config.num_classes = len(label_field.vocab) - 1
        config.kernel_sizes = [int(k) for k in config.kernel_sizes.split(',')]


        self.config = config

        vocab_size = config.vocab_size
        embedding_dim = config.embedding_dim
        num_classes = config.num_classes

        channel_in = 1
        channel_out = config.channel_out
        kernel_sizes = config.kernel_sizes

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = nn.ModuleList([nn.Conv2d(channel_in, channel_out,
                                              (k, embedding_dim))
                                    for k in kernel_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(len(kernel_sizes) * channel_out, num_classes)

    def forward(self, x):
        # print("input:", x.shape)
        # input: (batch, vocab_size)
        h = self.embedding(x) # (batch, vocab_size, embedding_dim)
        if self.config.static:
            h = h.to(device)

        h = h.unsqueeze(1) # (batch, 1, vocab_size, embedding_dim)
        h = [F.relu(conv(h)).squeeze(3) for conv in self.conv1] # [(batch, channel_out, vocab_size)] * len(kernel_sizes)
        h = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in h]  # [(batch, channel_out)] * len(kernel_sizes)
        h = torch.cat(h, 1) # (batch, channel_out * len(kernel_sizes)
        h = self.dropout(h)
        logit = self.fc1(h) # (batch, num_classes)
        return logit

# if __name__ == "__main__":
#     print("Loading data...")
#     text_field = data.Field(lower=True)
#     label_field = data.Field(sequential=False)
#     train_iter, dev_iter = get_MR(text_field, label_field,
#                                   device=-1, repeat=False,
#                                   batch_size=10)
#     print("Prepare data complete!")
#
#     model = CNNTextClassifier(text_field, label_field)
#     criterion = nn.CrossEntropyLoss()
#
#     for batch in train_iter:
#         feature, target = batch.text, batch.label
#         feature = feature.permute(1, 0) # batch first
#         target.data = target.data.sub(1) # index align
#
#         print("input shape:", feature.shape)
#         print("target shape:", target.shape)
#
#         logits = model(feature)
#
#         print("output logits shape:", logits.shape)
#
#         loss = criterion(logits, target)
#
#         print("loss:", loss.item())
#         break



