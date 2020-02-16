"""Implementation of pair filters"""
from typing import Sequence, Tuple

from co_bot.preprocessing.interfaces import IPairFilter


class IsNotTooLong(IPairFilter):
    """
    Filter used to remove pairs where at least one of the two sequences has more than _max_length characters

    Attributes:
        _max_length (str): maximum number of characters allowed in a sequence
    """
    def __init__(self, max_length):
        self._max_lenght = max_length

    def apply(self, pair: Tuple[str, str]) -> bool:
        """See parent method"""
        if len(pair[0]) > self._max_lenght  or len(pair[1]) > self._max_lenght:
            return False
        else:
            return True

    def apply_on_pairs(self, pairs: Sequence[Tuple[str, str]]) -> Sequence[Tuple[str, str]]:
        """See parent method"""
        filtered_pairs = []
        for pair in pairs:
            if self.apply(pair):
                filtered_pairs.append(pair)
            else:
                pass
        return filtered_pairs