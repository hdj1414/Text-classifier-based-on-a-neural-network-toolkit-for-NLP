import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class BaseModel(nn.Module):
    def __init__(self, args, vocab, tag_size):
        super(BaseModel, self).__init__()
        self.args = args
        self.vocab = vocab
        self.tag_size = tag_size

    def save(self, path):
        print(f'Saving model to {path}')
        ckpt = {
            'args': self.args,
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        torch.save(ckpt, path)

    def load(self, path):
        print(f'Loading model from {path}')
        ckpt = torch.load(path)
        self.vocab = ckpt['vocab']
        self.args = ckpt['args']
        self.load_state_dict(ckpt['state_dict'])

def load_embedding(vocab, emb_file="glove.6B.300d.txt", emb_size=300):
    embedding_matrix = np.zeros((len(vocab), emb_size))
    with open(emb_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            if word in vocab:
                idx = vocab[word]
                vector = np.asarray(values[1:], dtype='float32')
                embedding_matrix[idx] = vector
    return embedding_matrix

class DanModel(BaseModel):
    def __init__(self, args, vocab, tag_size):
        super(DanModel, self).__init__(args, vocab, tag_size)
        self.define_model_parameters()
        self.init_model_parameters()

        if args.emb_file is not None:
            self.copy_embedding_from_numpy()

    def define_model_parameters(self):
        self.embedding = nn.Embedding(len(self.vocab), self.args.emb_size)
        self.dropout_emb = nn.Dropout(self.args.emb_drop)
        
        self.fc_layers = nn.ModuleList()
        for i in range(self.args.hid_layer):
            input_size = self.args.emb_size if i == 0 else self.args.hid_size
            self.fc_layers.append(nn.Linear(input_size, self.args.hid_size))
            self.fc_layers.append(nn.ReLU())
            self.fc_layers.append(nn.Dropout(self.args.hid_drop))
            
        self.classifier = nn.Linear(self.args.hid_size, self.tag_size)

    def init_model_parameters(self, v=0.08):
        for name, param in self.named_parameters():
            nn.init.uniform_(param, -v, v)

    def copy_embedding_from_numpy(self, embedding_matrix):
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.data[self.vocab['<pad>']] = torch.zeros(self.args.emb_size)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.dropout_emb(embedded)
        
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1)
        
        for layer in self.fc_layers:
            if isinstance(layer, nn.Linear):
                pooled = layer(pooled)
            elif isinstance(layer, nn.ReLU):
                pooled = F.relu(pooled)
            elif isinstance(layer, nn.Dropout):
                pooled = layer(pooled)
        
        scores = self.classifier(pooled)
        return scores
