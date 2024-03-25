from collections.abc import Callable, Iterable, Mapping
from PIL import Image
import numpy as np
import os
from typing import Any, List, Union, Tuple, Dict
from hashlib import md5
from abc import abstractmethod, ABC
import sys
from threading import Thread

def get_file_name(path:str)->str:
    filepath, filename =os.path.split(path)
    name, suffix = os.path.splitext(filename)
    return name

class Calculate(ABC):
    @abstractmethod
    def _cal(self, panel_data:Tuple[np.array, np.array, np.array])->np.array:
        NotImplementedError
    
    class CalThread(Thread):
        def __init__(self, group: None = None, target: Callable[..., object] | None = None, name: str | None = None, args: Iterable[Any] = ..., kwargs: Mapping[str, Any] | None = None, *, daemon: bool | None = None) -> None:
            super().__init__(group, target, name, args, kwargs, daemon=daemon)
            self.result = None
            self.func = target
            self.args = args
        def run(self) -> None:
            self.result = self.func(self.args)
        def get_result(self):
            return self.result
    

class ImageBase:
    """
    the base class for the compression class
    """
    def __init__(self, image_path:Union[list, str], output:str) -> None:
        self.image_data = {}
        self.k = 0
        self.image_path = image_path
        self.output = output
        self.type = "jpg" # jpg, png, bmp
        self.usehash = True
    
    def _cal(self,panel_data:Tuple[np.array, np.array, np.array])->np.array:
        '''
        the interface to realize the compression method

        :params panel_data: the original image panel data
        :return: compression data
        '''
        NotImplementedError

    def seperation(self, image_data:np.array)->Tuple[np.array, np.array, np.array]:
        '''
        divide the image full color data into three panel data

        :params image_data: original full color data
        :return: tuple that contains three panel data 
        '''
        red = image_data[:,:,0]
        green = image_data[:,:,1]
        blue = image_data[:,:,2]
        return (red, green, blue)

    def __check_path(self, path:Union[list, str])->None:
        '''
        check the path is existed or not

        :params path: the absolute path that may be list or just a string path
        '''
        if isinstance(path, str):
            if not os.path.isabs(path):
                path = os.path.abspath(path)
            if not os.path.exists(path):
                raise FileNotFoundError(f"The file {path} does not exist")
            return
        for p in path:
            if not os.path.exists(p):
                raise FileNotFoundError(f"The file {path} does not exist")

    def compress(self) -> np.array:
        '''
        compress the image data using alogrithm
        :params image_data: original full color image data
        :params k: number of singular values to be kept
        :return: compressed image data
        '''
        self.read_image()

        for k,v in self.image_data.items():
            panel_data = self.seperation(v)
            self.image_data[k] = self._cal(panel_data)
        
        self.save()

    def save(self)->bool:
        '''
        save the compressed image data
        '''
        self.__check_path(self.output)
        for k,v in self.image_data.items():
            # try:
            zeros = np.zeros((v.shape[0], v.shape[1], 3), dtype=np.uint8)
            image = Image.fromarray(np.uint8(v.real + zeros))
            output_path = os.path.join(self.output, f"{k}.{self.type}")
            image.save(output_path)
            # except:
            #     raise Exception(f"Cannot save the file")

    def read_image(self)->np.array:
        '''
        load the image data
        '''
        self.__check_path(self.image_path)
        if isinstance(self.image_path, str):
            image = Image.open(self.image_path)
            image_array = np.array(image)
            self.image_data[md5(self.image_path.encode('utf-8')).hexdigest() if self.usehash else get_file_name(self.image_path)] = image_array
            return
        for p in self.image_path:
            image = Image.open(p)
            image_array = np.array(image)
            self.image_data[md5(p.encode('utf-8')).hexdigest() if self.usehash else get_file_name(self.image_path)] = image_array