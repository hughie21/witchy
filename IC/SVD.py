import numpy as np
import os
from typing import List, Union, Tuple, Dict
from .base import ImageBase, Calculate
            

class SVD_compressor(ImageBase, Calculate):
    """
    use the SVD method the compress the image data

    examples::

    >>> compression = SVD_compressor(r'image.jpg', './output/')
    >>> compression.k = 1000 # set the rank of characteristic matrix
    >>> compression.compress()
    """
    def __init__(self, image_path: list | str, output:str):
        super().__init__(image_path, output)
    
    def SVD(self,data:np.array)->Tuple[np.array, np.array, np.array]:
        '''
        SVD decomposition algorithm
        '''
        U,S,V = np.linalg.svd(data)
        if self.k > S.shape[0]:
            raise Exception(f"k is too large, the maximum of k is {S.shape[0]}")
        zeros = np.zeros((self.k,self.k))
        _s = S[:self.k]
        for i in range(self.k):
            zeros[i,i] = _s[i]
        return (U,zeros,V)

    def _cal(self,panel_data:Tuple[np.array, np.array, np.array])->np.array:
        '''
        calculate the three panel data's SVD matrix

        :params panel_data: a list of panel data
        :return: return the compressed image data
        '''
        red, green, blue = panel_data
        U0, S0, V0 = self.SVD(red)
        U1, S1, V1 = self.SVD(green)
        U2, S2, V2 = self.SVD(blue)
        C0 = np.dot(np.dot(U0[:,:self.k],S0),V0[:self.k,:])
        C1 = np.dot(np.dot(U1[:,:self.k],S1),V1[:self.k,:])
        C2 = np.dot(np.dot(U2[:,:self.k],S2),V2[:self.k,:])
        return np.stack((C0,C1,C2),axis=2)

    

