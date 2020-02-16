"""Definition of the Bahdanau Decoder"""
from torch.nn import GRU, Embedding, Module, Linear, Softmax, Parameter
from torch import Tensor, load, tanh, FloatTensor, bmm


class BahdanauDecoder(Module):
    """Decoder with Bahdanau attention mechanism

    Attributes:
        hidden_size (int): Number of features in the hidden state (used in the RNN and in the attention mechanism)
        nb_layers (int): Number of recurrent layers in the RNN
        vocabulary_size (int): Number of words in the vocabulary used to produce the output of the decoder
        dropout (float): If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer,
            with dropout probability equal to dropout (default=0)

    """

    def __init__(self, hidden_size: int, nb_layers:int, vocabulary_size: int, dropout: int = 0):
        super().__init__()
        self._hidden_size = hidden_size
        self._embedding = Embedding(vocabulary_size, self._hidden_size)
        self._fc_hidden = Linear(self._hidden_size, self._hidden_size)
        self._fc_encoder = Linear(self._hidden_size, self._hidden_size)
        self._weight = Parameter(FloatTensor(self._hidden_size, 1))
        self._softmax = Softmax(dim=2)
        self._rnn = GRU(input_size=self._hidden_size,
                        hidden_size=self._hidden_size,
                        num_layers=nb_layers,
                        bidirectional=False,
                        dropout=(0 if nb_layers == 1 else dropout),
                        batch_first=False)
        self._classifier = Linear(self._hidden_size, vocabulary_size)

    def forward(self, input_token: Tensor, decoder_hidden: Tensor, encoder_output: Tensor):
        """Forward pass of the decoder

        Args:
            input_token (Tensor): tensor of shape (1, B) that contains the input token for each element in the batch
                where B is the size of the batch
            decoder_hidden (Tensor): previous hidden state of the decoder that is a tensor of shape
                (num_layers*num_directions, B, H) with H the hidden_size
            encoder_output (Tensor): output of the encoder that is a tensor of shape (T, B, H) where T the length of
                the longest encoder input sequence in the batch

        """
        # Compute the embedding of the decoder input token
        decoder_embedded = self._embedding(input_token)

        # Decoder hidden tensor and encoder output tensor are passed through linear layers (fully connected)
        decoder_hidden = self._fc_hidden(decoder_hidden[0])
        encoder_output = self._fc_encoder(encoder_output)

        # Concat decoder hidden tensor and encoder output tensor then apply tanh function
        combined = tanh(decoder_hidden + encoder_output)

        # Compute attention weights
        alignement_weights = bmm(combined.transpose(0,1), self._weight.expand(10, self._hidden_size, 1))
        attention_weights = self._softmax(alignement_weights.transpose(1,2))

        # Compute the context vector
        context_vector = bmm(attention_weights, encoder_output.transpose(0,1))

        # Compute the output and the new hidden state of the RNN
        decoder_input = decoder_embedded.transpose(0,1) + context_vector
        decoder_output, decoder_hidden = self._rnn(decoder_input.transpose(0,1), decoder_hidden.unsqueeze(0))

        # Compute the output of the decoder (probability associated to each token in the vocabulary)
        decoder_output = self._softmax(self._classifier(decoder_output))

        return decoder_output, decoder_hidden


if __name__ == '__main__':
    # For testing purposes, to be removed
    encoder_output = load('../../data/output.pt')
    encoder_hidden = load('../../data/hidden.pt')

    print(encoder_output.shape)
    print(encoder_hidden.shape)

    ft = FloatTensor(1, 10, 50)

    bahdanau = BahdanauDecoder(50, 1, 10)
    bahdanau.forward(ft, encoder_hidden, encoder_output)