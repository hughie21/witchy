import os
import sys
import cv2
from typing import List, Union, Tuple, Dict
import numpy as np
from numba import jit
from concurrent.futures import ThreadPoolExecutor

import numpy as np
sys.path.append(os.getcwd())
from IC.base import ImageBase

"""
rewrite the ImageBase from the IC module, make it accept image data directly and redefine the compress method
"""

@jit(nopython=True)
def _dot(A: np.ndarray, B: np.ndarray, k:int)->np.array:
    mat_a = A[:, :k]
    mat_b = B.T[:k, :]
    mat_a = np.ascontiguousarray(mat_a)
    mat_b = np.ascontiguousarray(mat_b)
    _ = np.dot(mat_a, mat_b)
    return _

class Image_(ImageBase):
    def __init__(self):
        self.k = 0

    def compress(self, image_data:np.ndarray) -> np.array:
        panel_data = self.seperation(image_data)
        image_data = self._cal(panel_data)
        return image_data

class SVD_compressor(Image_):
    def __init__(self):
        super().__init__()

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

class PCA_compressor(Image_):
    def __init__(self):
        super().__init__()

    def PCA(self,data:np.array)->np.array:
        '''
        calculate the image data by PCA and use the selected k number of characteristic to compress
        :params data: the original image data
        :return: compressed data
        '''
        mean = np.mean(data, axis=0)
        sds = np.std(data, axis=0).reshape(1,-1)
        data = (data-mean)/sds
        data_T = data.T
        COV = np.matmul(data_T, data)
        W,Q = np.linalg.eig(COV)
        w_args = np.flip(np.argsort(W))
        Q = Q[:, w_args]
        W = W[w_args]
        C = np.matmul(data, Q)
        new_image_data = _dot(C, Q, self.k)
        new_image_data = new_image_data * sds + mean
        return new_image_data
    
    def _cal(self, panel_data:Tuple[np.array, np.array, np.array])->np.array:
        '''
        calculate the three panel data's PCA matrix

        :params panel_data: a list of panel data
        :return: return the compressed image data
        '''
        red, green, blue = panel_data
        tp = ThreadPoolExecutor(max_workers=3)
        z0 = tp.submit(self.PCA,red)
        z1 = tp.submit(self.PCA,green)
        z2 = tp.submit(self.PCA,blue)
        t0 = z0.result()
        t1 = z1.result()
        t2 = z2.result()
        tp.shutdown()
        return np.stack((t0, t1, t2), axis=2)

class DTC_compressor(Image_):
    def __init__(self)->None:
        super().__init__()
    
    def DTC(self, image_data: np.array) -> np.array:
        '''
        using the opencv's dtc method to compress the image data

        :params image_data: the original image data
        :return: the compressed image data
        '''
        image_data = image_data.astype('float32')
        c = cv2.dct(image_data)
        e = np.log(abs(c))
        image_rect = cv2.idct(e)
        temp = c[:self.k, :self.k]
        temp_2 = np.zeros(image_data.shape)
        temp_2[:self.k, :self.k] = temp
        _c = temp_2.astype('float32')
        return cv2.idct(_c)

    def _cal(self, panel_data:Tuple[np.array, np.array, np.array])->np.array:
        """
        calculate the three panel data's DTC matrix

        :params panel_data: a list of panel data
        :return: return the compressed image data
        """
        red, green, blue = panel_data
        d0 = self.DTC(red)
        d1 = self.DTC(green)
        d2 = self.DTC(blue)
        return np.stack((d0, d1, d2), axis=2)
