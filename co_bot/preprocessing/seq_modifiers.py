"""Implementation of sequence modifiers"""
from typing import Sequence
import re

from co_bot.preprocessing.interfaces import ISeqModifier


class Lowerize(ISeqModifier):
    """Modifier used to transform text into lower characters"""
    def apply(self, text_sequence: str) -> str:
        """See parent class"""
        return text_sequence.lower().strip()

    def apply_on_sequences(self, text_sequences: Sequence[str]) -> Sequence[str]:
        """See parent class"""
        modified_sequences = []
        for text_sequence in text_sequences:
            modified_sequences.append(self.apply(text_sequence))
        return modified_sequences

class SeparateCharacters(ISeqModifier):
    """
    Modifier used to separate specific characters from others with a whitespace

    Attributes:
        _characters_to_separate (str): pattern of characters to separate from other characters with a whitespace

    """
    def __init__(self, characters_to_separate: str):
        self._characters_to_separate = characters_to_separate

    def apply(self, text_sequence: str) -> str:
        """See parent class"""
        return re.sub(r"([{}])".format(self._characters_to_separate), r" \1", text_sequence)

    def apply_on_sequences(self, text_sequences: Sequence[str]) -> Sequence[str]:
        """See parent class"""
        modified_sequences = []
        for text_sequence in text_sequences:
            modified_sequences.append(self.apply(text_sequence))
        return modified_sequences


class KeepOnlyCharacters(ISeqModifier):
    """
    Modifier used to keep only some characters

    Attributes:
        _characters_to_keep (str): regular expression of characters to keep (all other characters would be removed)
    """
    def __init__(self, characters_to_keep: str):
        self._characters_to_keep = characters_to_keep

    def apply(self, text_sequence: str) -> str:
        """See parent class"""
        return re.sub(r"[^{}]+".format(self._characters_to_keep), r" ", text_sequence)

    def apply_on_sequences(self, text_sequences: Sequence[str]) -> Sequence[str]:
        """See parent class"""
        modified_sequences = []
        for text_sequence in text_sequences:
            modified_sequences.append(self.apply(text_sequence))
        return modified_sequences


class ConvertToUTF8(ISeqModifier):
    """Modifier used to convert text to UTF-8"""
    def apply(self, text_sequence: str) -> str:
        """See parent class"""
        return text_sequence.encode('ascii', errors='ignore').decode('utf-8')

    def apply_on_sequences(self, text_sequences: Sequence[str]) -> Sequence[str]:
        """See parent method"""
        modified_sequences = []
        for text_sequence in text_sequences:
            modified_sequences.append(self.apply(text_sequence))
        return modified_sequences


if __name__ == '__main__':
    # For testing purposes, to be removed
    input_output = ("You ever figure out what that thing's for ?", " No, see, I'm trying this new screening thing. You know, I figure if I'm always answering the phone, people'll think I don't have a life. My god, Rodrigo never gets pinned. ")
    input = input_output[1]
    #text = ConvertToUTF8().apply(input)
    text = Lowerize().apply(input)
    text = SeparateCharacters(characters_to_separate=".!?'").apply(text)
    text = KeepOnlyCharacters(characters_to_keep="a-zA-Z.?!'").apply(text)
    print(text)
