import pytorch_lightning as pl
import torch
from torchtext import data
import spacy
from torchtext import datasets
import os
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

        

class Improved_RNN(pl.LightningModule):

    def __init__(self):
        super().__init__()

        
    def forward(self, text, text_lengths):

        embedded = self.dropout(self.embedding(text))        
        #pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu())
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        #unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        return self.fc(hidden)  


    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        
        predictions = self.forward(batch.text[0], batch.text[1]).squeeze(1)
        # print("SHAPE : ",predictions, batch.label)
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(predictions, batch.label)
        # Logging to TensorBoard by default
        self.log('train_loss', loss)
        return loss


    def validation_step(self, batch, batch_idx):
        
        predictions = self.forward(batch.text[0], batch.text[1]).squeeze(1)
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(predictions, batch.label)
        # Logging to TensorBoard by default
        self.log('val_loss', loss)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
        

    def preprocess_data_ImprovedRNN(self, device='cpu', BATCH_SIZE=64):

        # tokenize using 'spacy' and make include lengths true to use padded sequences
        text = data.Field(tokenize = 'spacy', include_lengths = True)
        labels = data.LabelField(dtype = torch.float)

        # download imdb data and split as train and test
        train_data, test_data = datasets.IMDB.splits(text, labels)

        # split train as train and val sets
        train_data, valid_data = train_data.split()

        # define the max number of words to use for embedding
        MAX_VOCAB_SIZE = 25_000

        # use GloVe pretrained embeddings
        text.build_vocab(train_data, max_size = MAX_VOCAB_SIZE, unk_init = torch.Tensor.normal_) #, vectors = "glove.6B.100d", unk_init = torch.Tensor.normal_)
        labels.build_vocab(train_data)

        train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data), 
        batch_size = BATCH_SIZE,
        sort_within_batch = True,
        device = device)

        # retrieve the embeddings from the field's vocab
        pretrained_embeddings = text.vocab.vectors

        return text, labels, train_data, test_data, valid_data, train_iterator, valid_iterator, test_iterator, pretrained_embeddings


    def create_model_ImprovedRNN(self, input_dim, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, pad_idx):
        
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx = pad_idx)        
        self.rnn = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout)        
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)