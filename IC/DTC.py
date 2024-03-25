import cv2
from typing import List, Union, Tuple, Dict
import numpy as np
from .base import ImageBase, Calculate

class DTC_compressor(ImageBase, Calculate):
    """
    compress the image data via DTC
    """
    def __init__(self, image_path: list | str, output: str) -> None:
        super().__init__(image_path, output)
    
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
    
