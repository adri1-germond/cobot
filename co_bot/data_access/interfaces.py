"""Definition of interfaces used for accessing input data"""
from abc import ABC, abstractmethod
from typing import Sequence, Tuple


class IDataset(ABC):
    """Interface that define a dataset"""

    @abstractmethod
    def load_data(self, *args, **kwargs) -> Sequence[Tuple[str, str]]:
        """Read raw input data and pre process them to create sequences of dialogues"""
        pass
