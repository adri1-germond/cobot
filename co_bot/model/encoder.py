"""Definition of the Encoder class"""
from typing import Sequence

from torch.nn import GRU, Embedding, Module
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import Tensor


class Encoder(Module):
    """Implements the Encoder module

    Attributes:
        nb_layers (int): Number of recurrent layers in the RNN
        hidden_size (int): Number of features in the hidden state of the RNN
        vocabulary_size (int): Number of words in the corpus vocabulary
        dropout (float): If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer,
            with dropout probability equal to dropout (default=0)

    """

    def __init__(self, nb_layers: int, hidden_size: int, vocabulary_size: int, dropout: int = 0):
        """Constructor of the Encoder"""
        super().__init__()
        self._nb_layers = nb_layers
        self._hidden_size = hidden_size
        self._embedding = Embedding(vocabulary_size, self._hidden_size)
        self._rnn = GRU(input_size= self._hidden_size,
                        hidden_size=self._hidden_size,
                        num_layers=self._nb_layers,
                        bidirectional=True,
                        dropout=(0 if self._nb_layers == 1 else dropout),
                        batch_first=False)

    def forward(self, input_sequences: Tensor, input_lengths: Sequence[int], hidden=None):
        """
        Forward pass of the encoder

        Args:
            input_sequences (Tensor): tensor of shape (T, B) where T is the length of the longest sequence in the
            batch and B the size of the batch
            input_lengths (Sequence[int]): sequence of length of each sequence in the batch of input data
            hidden (Tensor): tensor

        Returns:
            Tensor, Tensor: outputs of the RNN with shape (T, B, H),
            hidden vectors with shape (num_layers*num_directions, B, H) where H is the hidden_size

        """
        embedded = self._embedding(input_sequences)
        print(embedded.shape)
        embedded = pack_padded_sequence(embedded, lengths=input_lengths, batch_first=False)
        outputs, hidden = self._rnn(embedded, hidden)
        outputs, _ = pad_packed_sequence(outputs, batch_first=False)
        outputs = outputs[:, :, :self._hidden_size] + outputs[:, :, self._hidden_size:]
        print(outputs.shape)
        print(hidden.shape)

        return outputs, hidden
