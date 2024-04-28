import struct
from typing import overload
from PIL import Image
import re
from .util import *
from .pool import AutoPool
import numpy as np
from .base import *
from .logging import Log

class LSB(Steganorgraphy):
    """
    The LSB steganography algorithm is a simple method for embedding data into images.

    ## example::
    ### embeding
    >>> encoder = LSB(lag=0)
    >>> file = File('your_path_to_image.png') # must be the PNG or BMP format
    >>> wm = "hellow world"
    >>> encoder.embed(file=file, wm=wm, mode='text', out_name='output.png')

    ### extracting
    >>> decoder = LSB(lag=0)
    >>> embed_img = File('output.png')
    >>> wm_shape = 88 # the length of watermark's bits (8bits)
    >>> wm = decoder.extract(embed_img="test.png", wm_shape=32, mode='text')
    >>> print(wm) # hellow world
    """
    def __init__(self, lag=0):
        """
        :params lag: the offset of the hidden watermark
        """
        super(LSB).__init__()
        self.log = Log('LSB')
        self.log.info(f"The hidden information will offset {lag} bit")
        self.lag = lag

    def getLstBit(self,source):
        return bin(source)[-1]

    def __extract(self, image:object, length:int)->list:
        image = np.array(image)
        width, height = image.shape[:2]
        index = 0
        lstBit = []
        for i in range(self.lag, width):
            for j in range(self.lag, height):
                for chanel in range(3):
                    if index >= length:
                        return "".join(lstBit)
                    lstBit.append(self.getLstBit(image[i,j,chanel]))
                    index += 1

    def __hide(self, image:object, bits:str):
        a1 = 1
        a2 = -1
        length = len(bits)
        index = 0
        image = np.array(image)
        width, height = image.shape[:2]
        for i in range(self.lag, width):
            for j in range(self.lag, height):
                for chanel in range(3):
                    if index >= length:
                        return image
                    if self.getLstBit(image[i,j,chanel]) == bits[index]:
                        image[i,j,chanel] &= a2 # just to keep the style unchanged
                        index += 1
                    else:
                        image[i,j,chanel] ^= a1 # when the least bit is different, reversal it
                        index += 1


    def embed(self, file=None, wm=None, mode='img', out_name=None):
        '''
        embed the hidden information to the target image

        :params file: the target File object
        :params wm: the watermark File object or the string
        :params mode: the mode of the watermark, 'img' or 'text'
        :params out_name: the output name of the embed image
        :return: the embed image if out_name is None, otherwise return None
        '''
        assert file is not None or wm is not None, 'Either file or wm must be provided'
        assert mode in ['img', 'text'], 'Mode must be either "img" or "text"'
        image = Image.open(hex2bin(file.bdata.hex))
        if mode == 'img':
            hideBitArr = bin(int(wm.bdata[:].hex(), 16))[2:]
        elif mode == 'text':
            hideBitArr = encode(wm)
        self.log.info(message=f'embed {len(hideBitArr)} bits into image size: ({image.size[0]}, {image.size[1]})')
        embed_image = Image.fromarray(self.__hide(image ,hideBitArr))
        if out_name is None:
            return embed_image
        else:
            self.log.info(message=f'The embed image is located in: {out_name}')
            embed_image.save(out_name, 'PNG')
    
    def extract(self, filename=None, embed_img=None, wm_shape=None, out_wm_name=None, mode='img'):
        '''
        get the hidden watermark from the image

        :params filename: the target image path
        :params embed_img: the target image File object
        :params wm_shape: the length of the watermark's bits
        :params out_wm_name: the output name of the watermark image
        :params mode: the mode of the watermark, 'img' or 'text'
        :return: if the mode is 'text' return the string form of the watermark, otherwise return none
        '''
        def toPic(bits:str)->Image.Image:
            bits = int(bits,2).to_bytes((len(bits)+7)//8, 'big')
            with open(out_wm_name, "wb") as f:
                f.write(bits)
        assert embed_img is not None or filename is not None, 'filename or embed_img must be provided'
        assert mode in ['img', 'text'], 'Mode must be either "img" or "text"'
        assert wm_shape is not None, 'wm_shape must be provided'
        if filename is not None:
            if not os.path.exists(filename):
                self.log.error(message=f'{filename} does not exist')
                raise FileNotFoundError("File does not exist")
            image = Image.open(filename)
        elif filename == None:
            image = Image.open(hex2bin(embed_img.bdata.hex))
        self.log.info(message=f'Extract {wm_shape} bits from image size: ({image.size[0]}, {image.size[1]})')
        lstBit = self.__extract(image, wm_shape)
        if mode == "text":
            text = decode(lstBit)
            self.log.info(message=f'The watermark extracted is: {text}')
            return text
        elif mode == "img" and out_wm_name is not None:
            self.log.info(message=f'The watermark extracted is: {out_wm_name}')
            return toPic(lstBit)
