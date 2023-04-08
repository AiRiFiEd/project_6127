import logging
from ..static import LOGGER_PRINT_LEVEL
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(LOGGER_PRINT_LEVEL)

import torch.nn as nn

class BaseRNN(nn.Module):
    def __init__(self, vocab_size: int, max_len: int, hidden_size: int, 
                    input_dropout_perc: float, dropout_perc: float,
                    n_layers: int, rnn_cell: str) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.input_dropout_perc = input_dropout_perc
        self.dropout_perc = dropout_perc
        
        rnn_cell = rnn_cell.lower()

        if rnn_cell == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError('unsupported RNN cell: {rnn_cell}'.format(rnn_cell=rnn_cell))
        
        self.input_dropout = nn.Dropout(p=input_dropout_perc)

    def forward(self, *args, **kwargs) -> None:
        raise NotImplementedError()
    