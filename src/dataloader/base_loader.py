"""Base Dataloader Interface."""
from abc import ABC, abstractmethod

class BaseDatasetLoader(ABC):
    @abstractmethod
    def load_data(self):
        pass
