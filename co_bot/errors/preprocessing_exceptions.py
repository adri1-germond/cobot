"""Custom exceptions for COBOT preprocessing pipeline"""
from enum import IntEnum, auto


class PreprocessingErrorCodes(IntEnum):
    """
    List of error codes used in COBOT preprocessing pipeline

    TOKEN_NOT_IN_VOCABULARY: see TokenNotInVocabulary
    TOKEN_ID_NOT_IN_VOCABULARY : see TokenIdNotIn Vocabulary

    """
    TOKEN_NOT_IN_VOCABULARY = auto()
    TOKEN_ID_NOT_IN_VOCABULARY = auto()


class CobotPreprocessingError(Exception):
    """
    Exceptions that can be thrown by COBOT preprocessing pipeline

    Attributes:
        _code (ErrorCodes): error code

    Args:
        code (ErrorCodes): see _code

    """
    def __init__(self, code=-1):
        """Call parent class init and set error code to unknown value -1"""
        super().__init__()
        self._code = code

    @property
    def code(self) -> int:
        """get _code attribute"""
        return self._code

    def __str__(self):
        return "{0}|{1}|".format(__class__.__name__, self._code)


class TokenNotInVocabulary(CobotPreprocessingError):
    """
    Token does not exist in vocabulary

    Attributes:
        _token (str): textual representation of the token

    """
    def __init__(self, token: str):
        super().__init__(PreprocessingErrorCodes.TOKEN_NOT_IN_VOCABULARY)
        self._token = token

    def __str__(self):
        return (
            super().__str__()
            + "Token {0} does not exist in vocabulary".format(self._token)
        )


class TokenIdNotInVocabulary(CobotPreprocessingError):
    """
    Token identifier does not exist in vocabulary

    Attributes:
        _token (int): identifier of the token

    """
    def __init__(self, token_id: int):
        super().__init__(PreprocessingErrorCodes.TOKEN_ID_NOT_IN_VOCABULARY)
        self._token_id = token_id

    def __str__(self):
        return (
            super().__str__()
            + "Token id {0} does not exist in vocabulary".format(self._token_id)
        )

