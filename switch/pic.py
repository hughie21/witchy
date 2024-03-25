from typing import Union
from IC.base import ImageBase
from PIL import Image
from io import BytesIO
import numpy as np
import cv2

class Image_Data(ImageBase):
    def __init__(self,k) -> None:
        self.k = k
    
    def compress(self, image_data:np.ndarray) -> np.array:
        panel_data = self.seperation(image_data)
        image_data = self._cal(panel_data)
        return image_data

class DTC_compressor(Image_Data):
    def __init__(self,k)->None:
        super().__init__(k)
    
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

    def _cal(self, panel_data:tuple[np.array, np.array, np.array])->np.array:
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

class Image_Convert:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def convert_to_damaged_images(data:bytes, quality:int=100)->Image:
        image = Image.open(BytesIO(data))
        image_data = np.array(image)
        dtc = DTC_compressor(quality)
        cimage = Image.fromarray(dtc.compress(image_data).astype('uint8'))
        return cimage
    
    @staticmethod
    def convert_to_lossless_images(data:bytes)->Image:
        image = Image.open(BytesIO(data))
        return image
    
    