import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.backends import cudnn

import project_6127.data.preprocessing as preprocessing
import project_6127.static as static
from project_6127.model.encoder import EncoderRNN
from project_6127.model.decoder import DecoderRNNFB
from project_6127.model.networks import WritingEditingNetwork
from project_6127.model.utils import train_epoches

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(static.LOGGER_PRINT_LEVEL)

def setup_backend():
    logger.debug('setting up backend...')
    cudnn.benchmark = True
    logger.info('cudnn benchmark set to {cudnn_benchmark}'.format(cudnn_benchmark=cudnn.benchmark))
    config = static.Config()
    logger.info('random seed: {seed}'.format(seed = config.seed))
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():        
        torch.cuda.manual_seed(config.seed)
        config.cuda_is_available = True
    logger.info('cuda is {is_available}available'.format(is_available = 
        (not torch.cuda.is_available()) * 'not '
    ))
    logger.debug('setting up backend completed.')
    return config

def load_data(name: str = 'train') -> preprocessing.Data:
    filepath = os.path.join(
        static.DIRECTORY_DATA, '{name}.dat'.format(name=name)
    )
    data = preprocessing.Data(filepath)
    data.load()
    return data

def build_embedding(vocab_size, embedding_size, padding_idx,
                        pretrained = None):
    embedding = nn.Embedding(vocab_size, embedding_size, 
                    padding_idx = padding_idx)
    return embedding

def build_model(vocab_size, embedding_size, max_length_title, max_length_abstract,                    
                    encoder_title_cell, encoder_title_n_layers,
                    encoder_title_input_dropout_perc,
                    encoder_title_bidirectional,
                    encoder_cell, encoder_n_layers,
                    encoder_input_dropout_perc,
                    encoder_bidirectional,
                    decoder_cell, decoder_n_layers,
                    decoder_input_dropout_perc,
                    decoder_bidirectional,
                    decoder_dropout_perc) -> nn.Module:
    
    embedding = build_embedding(vocab_size, embedding_size, 0)

    encoder_title = EncoderRNN(
                        vocab_size, embedding, max_length_title, 
                        embedding_size, input_dropout_perc=encoder_title_input_dropout_perc,
                        n_layers=encoder_title_n_layers, 
                        bidirectional=encoder_title_bidirectional, 
                        rnn_cell=encoder_title_cell
                    )
    encoder = EncoderRNN(vocab_size, embedding, max_length_abstract, 
                         embedding_size, input_dropout_perc=encoder_input_dropout_perc, 
                         variable_lengths = False, n_layers=encoder_n_layers, 
                         bidirectional=encoder_bidirectional, 
                         rnn_cell=encoder_cell)
    decoder = DecoderRNNFB(vocab_size, embedding, max_length_abstract, 
                            embedding_size, sos_id=2, eos_id=1,
                            n_layers=decoder_n_layers, rnn_cell=decoder_cell, 
                            bidirectional=decoder_bidirectional,
                            input_dropout_perc=decoder_input_dropout_perc, 
                            dropout_perc=decoder_dropout_perc)
    model = WritingEditingNetwork(encoder_title, encoder, decoder)
    return model

if __name__ == '__main__':
    config = setup_backend()
    data = load_data('train')
    
    vectorizer = preprocessing.Vectorizer(
        max_words = config.max_words,
        min_frequency = config.min_freq,
        start_end_tokens = config.start_end_tokens,
        max_len = config.max_len
    )
    
    abstracts = preprocessing.Abstracts()
    abstracts.process_corpus(data, vectorizer)
    vocab_size = vectorizer.vocabulary_size

    model = build_model(
        vocab_size = vocab_size,
        embedding_size = config.embedding_size,
        max_length_title = abstracts.head_len,
        max_length_abstract = abstracts.abs_len,
        encoder_title_cell = config.cell,
        encoder_title_n_layers = config.nlayers,
        encoder_title_input_dropout_perc = config.dropout,
        encoder_title_bidirectional = config.bidirectional,
        encoder_cell = config.cell,
        encoder_n_layers = config.nlayers,
        encoder_input_dropout_perc = config.dropout,
        encoder_bidirectional = config.bidirectional,   
        decoder_cell = config.cell,
        decoder_n_layers = config.nlayers,
        decoder_input_dropout_perc = config.dropout,
        decoder_bidirectional = config.bidirectional,
        decoder_dropout_perc = config.dropout
    )

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    if config.cuda_is_available:
        model = model.cuda()
        criterion = criterion.cuda()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    train_epoches(dataset = abstracts, model = model, 
                    criterion = criterion, 
                    optimizer = optimizer,
                    vocab_size = vocab_size,                    
                    teacher_forcing_ratio = 1,
                    config = config)
    
    filepath_params = os.path.join(static.DIRECTORY_OUTPUT, 'params.pkl')
    torch.save(model.state_dict(), filepath_params)

