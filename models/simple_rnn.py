import pytorch_lightning as pl
import torch
from torchtext import data
import spacy
from torchtext import datasets
import os
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class RNN(pl.LightningModule):

    def __init__(self):
        
        super().__init__()

        
    def forward(self, x):

        x = torch.tensor(x, dtype=torch.long)

        embedded = self.embedding(x)
           
        output, hidden = self.rnn(embedded)

        output = self.fc(hidden[0].squeeze(0))

        output = self.output(output)

        return output

        

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        
        predictions = self.forward(batch.text).squeeze(1)
        # print("SHAPE : ",predictions, batch.label)
        criterion = nn.BCELoss()
        loss = criterion(predictions, batch.label)
        # Logging to TensorBoard by default
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        
        predictions = self.forward(batch.text).squeeze(1)
        criterion = nn.BCELoss()
        loss = criterion(predictions, batch.label)
        # Logging to TensorBoard by default
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def preprocess_data(self, device='cpu', BATCH_SIZE=64):
        text = data.Field(lower=True)#tokenize = 'spacy')
        labels = data.LabelField(dtype = torch.float)
        train_data, test_data = datasets.IMDB.splits(text, labels)

        train_data, valid_data = train_data.split()

        text.build_vocab(train_data, max_size = 20000)
        labels.build_vocab(train_data)

        train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data), 
        batch_size = BATCH_SIZE,
        device = device)

        return text, labels, train_data, test_data, valid_data, train_iterator, valid_iterator, test_iterator

    def create_model(self, input_dim, embedding_dim, hidden_dim, output_dim):
        
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        
        self.rnn = nn.LSTM(embedding_dim, hidden_dim)
        
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.output = nn.Softmax()



class CNN(pl.LightningModule):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
                 dropout, pad_idx):
        
        super().__init__()
                
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fs, embedding_dim)) 
                                    for fs in filter_sizes
                                    ])
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        
        #text = [sent len, batch size]
        
        #text = text.permute(1, 0)
                
        #text = [batch size, sent len]
        text = torch.tensor(text, dtype=torch.long, device='cuda')
        # text=text.to('cuda')
        embedded = self.embedding(text)

        #embedded = [batch size, sent len, emb dim]
        
        embedded = embedded.unsqueeze(1)
        
        #embedded = [batch size, 1, sent len, emb dim]
        
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
            
        #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
                
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        #pooled_n = [batch size, n_filters]
        
        cat = self.dropout(torch.cat(pooled, dim = 1))

        #cat = [batch size, n_filters * len(filter_sizes)]
            
        return self.fc(cat)

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        
        predictions = self.forward(batch.text).squeeze(1)
        # print("SHAPE : ",predictions, batch.label)
        criterion = nn.BCEWithLogitsLoss()
        # criterion = nn.BCELoss()
        loss = criterion(predictions, batch.label)
        # Logging to TensorBoard by default
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
