from abc import ABC, abstractmethod

class Steganorgraphy(ABC):
    @abstractmethod
    def embed(self, file=None, wm=None, mode='img', out_name=None):
        NotImplementedError
    
    @abstractmethod
    def extract(self, filename=None, embed_img=None, wm_shape=None, out_wm_name=None, mode='img'):
        NotImplementedError

class OutOfRangeError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
    def __str__(self) -> str:
        return f"{self.__class__.__name__}: {self.args[0]}"