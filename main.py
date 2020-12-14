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
import lime
from DependencyParsing.stanzaParser import *
# from xai.shap import xai_shap

# from captum.attr import LayerIntegratedGradients, TokenReferenceBase, visualization
# from xai.visualize import interpret_sentence
from models.improved_rnn import Improved_RNN

nlp = spacy.load('en')


torch.backends.cudnn.deterministic = True
device='cpu'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEED = 0

torch.manual_seed(SEED)


# restore_from_checkpoint = 0

# #call model
# model = RNN()

# #get data
# text, labels, train_data, test_data, valid_data, train_iterator, valid_iterator, test_iterator = model.preprocess_data(device, BATCH_SIZE=64)

# # init model
# INPUT_DIM = len(text.vocab)
# EMBEDDING_DIM = 100
# HIDDEN_DIM = 256
# OUTPUT_DIM = 1

# model.create_model(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)

# print("Success")

# if(restore_from_checkpoint):
#     # PATH = './lightning_logs/version_1/checkpoints/epoch=14.ckpt'
#     # hparams_path = './lightning_logs/version_1/hparams.yaml'
#     #reload trined module
#     model.load_state_dict(torch.load("./models/model_imdb.pt"))
#     #print(model.learning_rate)
#     # prints the learning_rate you used in this checkpoint
#     model.eval()

# else:
#     # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
#     # trainer = pl.Trainer(gpus=8) (if you have GPUs)
#     trainer = pl.Trainer(max_epochs=15, gpus=0)
#     trainer.fit(model,  train_iterator)
#     torch.save(model.state_dict(), ".models/model_imdb.pt")

# print('Model loaded')
# #shap

#convert data to integers
# x_train_int = [vars(entry).get('text') for entry in train_data[:50]]
# x_train_int = text.pad(x_train_int)
# x_train_int = text.numericalize(x_train_int)
# x_test_int = [vars(entry).get('text') for entry in test_data[:50]]
# x_test_int = text.pad(x_test_int)
# x_test_int = text.numericalize(x_test_int)

# print((model))

# xai_shap(model, x_train_int, x_test_int)


from torchtext import vocab
import torchtext
 

#loaded_vectors = vocab.GloVe(name='6B', dim=50)
TEXT = torchtext.data.Field(lower=True, tokenize='spacy', batch_first=True)
Label = torchtext.data.LabelField(dtype = torch.float)
# train, test = torchtext.datasets.IMDB.splits(text_field=TEXT,
#                                       label_field=Label,
#                                       train='train',
#                                       test='test'
# )
#                                       #path='data/aclImdb')
# test= test.split(split_ratio = 0.04)
train_data, test_data = datasets.IMDB.splits(TEXT, Label)

train_data, valid_data = train_data.split()
MAX_VOCAB_SIZE = 25000
# If you prefer to use pre-downloaded glove vectors, you can load them with the following two command line
#loaded_vectors = torchtext.vocab.Vectors('C:/Users/Abhishek Saroha/Documents/GitHub/beyond-simple-word-level-input-relevance-explanations/data/glove.6B.50d.txt')
#TEXT.build_vocab(train_data, vectors=loaded_vectors, max_size=len(loaded_vectors.stoi))
#TEXT.build_vocab(train_data, max_size = MAX_VOCAB_SIZE, vectors = "glove.6B.100d", unk_init = torch.Tensor.normal_)
TEXT.build_vocab(train_data, max_size = MAX_VOCAB_SIZE, unk_init = torch.Tensor.normal_)

 
#TEXT.vocab.set_vectors(stoi=loaded_vectors.stoi, vectors=loaded_vectors.vectors, dim=loaded_vectors.dim)
Label.build_vocab(train_data)

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data), 
        batch_size = 64,
        device = device)

# PAD_IND = TEXT.vocab.stoi['pad']

# token_reference = TokenReferenceBase(reference_token_idx=PAD_IND)

INPUT_DIM = len(TEXT.vocab)
print("########## ",INPUT_DIM)
EMBEDDING_DIM = 100
N_FILTERS = 100
FILTER_SIZES = [3,4,5]
OUTPUT_DIM = 1
DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)

root_words = root_generator()
child_words = child_generator()

#trainer = pl.Trainer(max_epochs=5, gpus=1)
#trainer.fit(model, train_iterator)
#torch.save(model.state_dict(), "./models/model_cnn_self_imdb.pt")

model.load_state_dict(torch.load('models/imdb-model-cnn.pt'))
model.eval()


predict_sentiment_root =  model(root_words)

print(predict_sentiment_root)


model = model.to(device) 