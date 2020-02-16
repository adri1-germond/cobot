"""Definition of modifiers/filters preprocessing chains"""
from typing import Sequence, Tuple

from co_bot.preprocessing.interfaces import ISeqModifier, IPairFilter


class TextSeqChain:
    """
    Pipeline used to sequentially apply modifiers on text sequences

    Attributes:
        _modifiers (Sequence[ISeqModifier]): sequence of modifiers

    """
    def __init__(self, modifiers: Sequence[ISeqModifier] = ()):
        self._modifiers = modifiers

    def apply(self, text_sequence: str):
        """
        Apply the pipeline of modifiers to input text sequence

        Args:
            text_sequence (str): input text sequence

        Returns:
            str : text sequence after applying modifiers
        """
        for modifier in self._modifiers:
            text_sequence = modifier.apply(text_sequence)

        return text_sequence

    def apply_on_sequences(self, text_sequences: Sequence[str]):
        """
        Apply the pipeline of modifiers to input text sequences

        Args:
            text_sequences[Sequence[str]]: input

        Returns:
            Sequence[str]: text sequences after applying modifiers

        """
        for modifier in self._modifiers:
                text_sequences = modifier.apply_on_sequences(text_sequences)

        return text_sequences


class PairChain:
    """
    Pipeline used to sequentially apply filters on text sequences pairs

    Attributes:
        _filters (Sequence[IPairFilter]): sequence of filters

    """
    def __init__(self, filters: Sequence[IPairFilter] = ()):
        self._filters = filters

    def apply(self, pair: Tuple[str, str]) -> bool:
        """
        Apply the pipeline of filters to input pairs of text sequences

        Args:
            pair (Tuple[str, str]): input pair of text sequences

        Returns:
            bool: text sequence after applying modifiers
        """
        for filter in self._filters:
            if not filter.apply(pair):
                return False
        return True

    def apply_on_pairs(self, pairs: Sequence[Tuple[str, str]]) -> Sequence[Tuple[str, str]]:
        """
        Apply the pipeline of filters to input pairs of text sequences

        Args:
            pairs[Sequence[Tuple[str, str]]]: input pairs of text sequences

        Returns:
            Sequence[str]: text sequences after applying modifiers

        """
        for filter in self._filters:
            pairs = filter.apply_on_pairs(pairs)

        return pairs

