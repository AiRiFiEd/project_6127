import logging
from ..static import LOGGER_PRINT_LEVEL
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(LOGGER_PRINT_LEVEL)

import torch.nn as nn

from .base import BaseRNN

class EncoderRNN(BaseRNN):
    def __init__(self, vocab_size: int, embedding: nn.Embedding,
                    max_len: int, hidden_size: int, 
                    input_dropout_perc: float = 0.,
                    dropout_perc: float = 0., n_layers: int = 1, 
                    bidirectional: bool = False, rnn_cell: str = 'gru',
                    variable_lengths: bool = True) -> None:
        super().__init__(vocab_size=vocab_size, 
                            max_len=max_len,
                            hidden_size=hidden_size,
                            input_dropout_perc=input_dropout_perc,
                            dropout_perc=dropout_perc,
                            n_layers=n_layers, rnn_cell=rnn_cell)
        self.embedding = embedding
        self.variable_lengths = variable_lengths
        self.rnn = self.rnn_cell(
            hidden_size, hidden_size, n_layers,
            batch_first=True, bidirectional=bidirectional, 
            dropout=dropout_perc
        )

    def forward(self, input_var, input_lengths=None):
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)
        
        if self.variable_lengths:            
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, 
                                input_lengths, batch_first=True)
        output, hidden = self.rnn(embedded)
        if self.variable_lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        return output, hidden