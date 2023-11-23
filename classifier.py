import torch
import torch.nn as nn


class ToxicityClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_size):
        super(ToxicityClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.fc = nn.Linear(embedding_dim, output_size)


    def forward(self, x):
        x = self.embedding(x)
        x = x.mean(dim=1)  # Média sobre as dimensões do embedding
        x = self.fc(x)
        return torch.sigmoid(x)