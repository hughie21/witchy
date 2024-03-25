import numpy as np
from typing import List, Union, Tuple, Dict
from .base import ImageBase, Calculate
from numba import jit

@jit
def _dot(A: np.ndarray, B: np.ndarray, k:int)->np.array:
    mat_a = A[:, :k]
    mat_b = B.T[:k, :]
    mat_a = np.ascontiguousarray(mat_a)
    mat_b = np.ascontiguousarray(mat_b)
    _ = np.dot(mat_a, mat_b)
    return _

class PCA_compressor(ImageBase, Calculate):
    """
    use the PCA alogrithm to compress the image
    """
    def __init__(self, image_path: list | str, output:str):
        super().__init__(image_path, output)
    
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
        COV = np.dot(data_T, data)
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
        thread_pool = []
        for d in panel_data:
            thread_pool.append(self.CalThread(target=self.PCA, args=d))

        for t in thread_pool:
            t.start()
        
        for t in thread_pool:
            t.join()
        
        r0 = thread_pool[0].get_result()
        r1 = thread_pool[1].get_result()
        r2 = thread_pool[2].get_result()
        # print("")
        return np.stack((r0,r1,r2), axis=2)


