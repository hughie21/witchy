from .PCA import PCA_compressor
from .SVD import SVD_compressor
from .DTC import DTC_compressor
from typing import List, Union, Tuple, Dict
import numpy as np

class Image_compressor:
    """
    the main program of the library, intergrate the alogrithm

    example::
    >>> compressor = Image_compressor('practise.jpg','./')
    >>> compressor.compress(method='svd',k=1000, usehash=True)
    """
    def __init__(self, image_path:Union[list, str], output:str) -> None:
        '''
        :params image_path: the path of the image can be a string or a list of path
        :params output: the path of the outputing image
        '''
        self.compressor = {
            "svd": SVD_compressor(image_path, output),
            "pca": PCA_compressor(image_path, output),
            "dtc": DTC_compressor(image_path, output),
        }
    
    def compress(self, method, k:int=100, usehash=True):
        '''
        the method the will call the different alogrithm to compress the image

        :params method: the alogrithm name, supported name including: pca, svd, dtc
        :params k: the charateristics that will be keeped. The smaller it is, the image will be more smaller as well, yet the image quantity may be terrible
        :params usehash: it determine if the output file name used by hash(md5), if it is false it will still use its original name
        '''
        if method == None:
            raise ValueError("Method is not specified")
        if method not in self.compressor.keys():
            raise ValueError(f"{method} is not supported")

        self.compressor[method].k = k
        self.compressor[method].usehash = usehash
        self.compressor[method].compress()
        
