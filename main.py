import pytorch_lightning as pl
import torch
from torchtext import data
import spacy
from torchtext import datasets
import os
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.simple_rnn import RNN, CNN
from models.improved_rnn import Improved_RNN

nlp = spacy.load('en')


torch.backends.cudnn.deterministic = True
#device='cpu'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEED = 0

torch.manual_seed(SEED)


restore_from_checkpoint = 0

#call RNN() model
#model = RNN()

#call Improved_RNN() model
model = Improved_RNN()

#get data for RNN()
#text, labels, train_data, test_data, valid_data, train_iterator, valid_iterator, test_iterator = model.preprocess_data(device, BATCH_SIZE=64)

#get data for Improved_RNN()
text, labels, train_data, test_data, valid_data, train_iterator, valid_iterator, test_iterator, pretrained_embeddings = model.preprocess_data_ImprovedRNN(device, BATCH_SIZE=64)


# init model 
INPUT_DIM = len(text.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
PAD_IDX = text.vocab.stoi[text.pad_token]
UNK_IDX = text.vocab.stoi[text.unk_token]

# create model for RNN()
#model.create_model(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)

# create model for Improved_RNN()
model.create_model_ImprovedRNN(INPUT_DIM,
                               EMBEDDING_DIM, 
                               HIDDEN_DIM, 
                               OUTPUT_DIM, 
                               N_LAYERS, 
                               BIDIRECTIONAL, 
                               DROPOUT, 
                               PAD_IDX) 
                               
# replace the initial weights of the embedding layer with the pre-trained embeddings
#model.embedding.weight.data.copy_(pretrained_embeddings)

# initialize <unk> and <pad> as zeros to explicitly tell our model that they are irrelevant for determining sentiment
model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

# use gpu
model = model.to(device)



print("Success")

if(restore_from_checkpoint):
    # PATH = './lightning_logs/version_1/checkpoints/epoch=14.ckpt'
    # hparams_path = './lightning_logs/version_1/hparams.yaml'
    #reload trined module
    model.load_state_dict(torch.load("./models/model_imdb.pt"))
    #print(model.learning_rate)
    # prints the learning_rate you used in this checkpoint
    model.eval()

else:
    # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
    # trainer = pl.Trainer(gpus=8) (if you have GPUs)
    trainer = pl.Trainer(max_epochs=5, gpus=1)
    trainer.fit(model,  train_iterator)
    torch.save(model.state_dict(), ".models/model_imdb.pt")



