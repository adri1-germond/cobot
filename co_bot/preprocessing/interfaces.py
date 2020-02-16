"""Definition of interfaces used in preprocessing pipeline"""
from abc import ABC, abstractmethod
from typing import Tuple, Sequence


class ISeqModifier(ABC):
    """Interface that define a text modifier"""

    @abstractmethod
    def apply(self, text_sequence: str) -> str:
        """
        Apply the modifier on the input text sequence

        Args:
            text_sequence: input text sequence to be modified
        """

    @abstractmethod
    def apply_on_sequences(self, text_sequences: Sequence[str]):
        """
        Apply the modifier on the input text sequences

        Args:
            text_sequences: input text sequences to be modified
        """


class IPairFilter(ABC):
    """Interface that define a pair of sequences filter"""

    @abstractmethod
    def apply(self, pair: Tuple[str, str]) -> bool:
        """
        Apply the filter on the input pair of sequences

        Args:
            pair (Tuple[str, str]): pair of sequences
        """

    @abstractmethod
    def apply_on_pairs(self, pairs: Sequence[Tuple[str, str]]) -> Sequence[Tuple[str, str]]:
        """
        Apply the filter on the input pairs of sequences

        Args:
            pairs (Sequence[Tuple[str, str]]): pairs of sequences
        """
