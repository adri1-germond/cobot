"""Definition of BatchBuilder"""
import random
from typing import Sequence, Tuple

from torch.nn.utils.rnn import pad_sequence
from torch import LongTensor, Tensor

from co_bot.preprocessing.tonekizer import Tokenizer


class BatchBuilder:
    """
    Used to create batches of training data

    Attributes:
        training_data (Sequence[Tuple[str, str]]): pairs of text sequences (input, output) that will be used as
        training data for the neural network
        _tokenizer (Tokenizer): Class used to tokenize text sequences

    """

    def __init__(self, training_data: Sequence[Tuple[str, str]], nb_epochs: int = 3):
        self._training_data = [(index, tuple(data[0]), tuple(data[1])) for index, data in enumerate(training_data)]
        self._nb_epochs = nb_epochs
        self._training_buffer = []
        self._current_epoch = 0

    def _initialize_buffer(self):
        """Initialize
        """
        self._training_buffer = self._training_data
        self._current_epoch += 1

    def get_batch(self, batch_size: int = 5) -> (Tensor, Sequence[int], Tensor, Sequence[int]):
        """
        Get a batch of training_data with at most batch_size elements in it

        A batch of data is built for a given epoch taking without replacement batch_size elements of training_data.
        If there are not enough remaining elements to make a batch with batch_size length, a shorter batch would be created.

        This method support batching data for multiple epochs. When data for all epochs have been returned, the method
        will return an empty list.

        Args:
            batch_size (int): maximum size of the batch

        Returns
            Sequence[Tuple[[Sequence[int], Sequence[int]]]]: batch of training data

        """
        if not self._training_buffer and self._current_epoch < self._nb_epochs:
            self._initialize_buffer()
        elif not self._training_buffer and self._current_epoch == self._nb_epochs:
            return Tensor([]), [], Tensor([]), []

        if len(self._training_buffer) < batch_size:
            corrected_batch_size = len(self._training_buffer)
        else:
            corrected_batch_size = batch_size
        batch = random.sample(self._training_buffer, corrected_batch_size)
        self._training_buffer = list(set(self._training_buffer) - set(batch))

        batch.sort(key=lambda x:len(x[1]), reverse=True)

        input_sequences = [pair[1] for pair in batch]
        output_sequences = [pair[2] for pair in batch]
        input_sequences, input_lengths = self._process_sequences(input_sequences)
        output_sequences, output_lengths = self._process_sequences(output_sequences)

        return input_sequences, input_lengths, output_sequences, output_lengths


    def _process_sequences(self, sequences: Sequence[Sequence[int]]) -> (Tensor, Sequence[int]):
        """
        Convert the input sequences into a tensor and apply zero padding

        Args:
            sequences (Sequence[Sequence[int]]): batch of sequences

        Returns:
          (Tensor, Sequence[int]): Tensor that contains batch of training data, sequence of lengths of training data

        """
        #sequences = [self._tokenizer.process_sentence(sequence) for sequence in sequences]
        sequences_lengths = [len(sequence) for sequence in sequences]
        tensor = [LongTensor(sequence) for sequence in sequences]
        tensor = pad_sequence(tensor, batch_first=False)

        return tensor, sequences_lengths
