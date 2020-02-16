"""Definition of Tokenizer"""
from typing import Dict, Sequence

from co_bot.errors.preprocessing_exceptions import TokenNotInVocabulary, TokenIdNotInVocabulary


class Tokenizer:
    """
    Used to tokenize sentences and hold vocabulary of the corpus

    Attributes:
        _vocabulary (Dict): mapping from a token to its numerical identifier
        _word_count (Dict): mapping from a token to its number of occurrences in the corpus
        _size_vocabulary (int): number of unique tokens in the corpus
        _indexes (Dict): mapping from a numerical identifier to its associated token
    """

    def __init__(self):
        self._vocabulary = {"SOS": 1,
                            "EOS": 2,
                            "PAD": 0}
        self._word_count = {}
        self._size_vocabulary = 3
        self._indexes = {1: "SOS",
                         2: "EOS",
                         0: "PAD"}

    def process_sentence(self, sentence: str) -> Sequence[int]:
        """
        Tokenize a sentence and get indexes of its tokens

        Args:
            sentence (str): input sentence

        Returns:
            Sequence[int]: indexes of tokens that constitute the sentence
        """
        tokenized_sentence = sentence.split()
        indexed_sentence = [self._vocabulary["SOS"]]
        for token in tokenized_sentence:
            self._add_token(token)
            indexed_sentence.append(self._vocabulary[token])
        indexed_sentence.append(self._vocabulary["EOS"])

        return indexed_sentence

    def _add_token(self, token: str):
        """
        Add a token in the vocabulary

        If the token has never been seen in the corpus, associate it to a unique numerical identifier then update
        the vocabulary and indexes mapping. Finally, set its word count to one.

        If the token has already been seen in the corpus, increment by one its word count.

        Args:
            token (str): textual representation of the token

        Returns:
            None
        """
        if token not in self._vocabulary:
            self._vocabulary[token] = self._size_vocabulary
            self._indexes[self._size_vocabulary] = token
            self._size_vocabulary += 1
            self._word_count[token] = 1
        else:
            self._word_count[token] += 1

    @property
    def vocabulary(self) -> Dict[str, int]:
        """
        Get the mapping between token and its index

        Returns:
            Dict[str, int]: mapping from a token to its index
        """
        return self._vocabulary

    @property
    def indexes(self) -> Dict[int, str]:
        """
        Get the mapping between an index and its token

        Returns:
            Dict[int, str]: mapping from an index to its token
        """
        return self._indexes

    @property
    def word_count(self) -> Dict:
        """
        Get the mapping between a token and its number of occurrences in the corpus

        Returns:
            Dict[str, int]: mapping from a token to its number of occurrences
        """
        return self._word_count

    def get_token_id(self, token: str) -> int:
        """
        Get token identifier

        Args:
            token (str): textual representation of the token

        Returns:
            int: identifier of the token in the vocabulary

        Raises:
            TokenNotInVocabulary: case that token not in the vocabulary
        """
        try:
            token_id = self._vocabulary[token]
            return token_id

        except KeyError as exception:
            raise TokenNotInVocabulary(token) from exception

    def get_token_from_id(self, identifier: int) -> str:
        """
        Get the token from its identifier

        Args:
            identifier (int): identifier of the token

        Returns:
            str: textual representation of the token

        Raises:
            TokenIdNotInVocabulary: case that token identifier not in the vocabulary
        """
        try:
            token = self._indexes[identifier]
            return token

        except KeyError as exception:
            raise TokenIdNotInVocabulary(identifier) from exception