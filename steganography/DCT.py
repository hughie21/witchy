import numpy as np
from numpy.linalg import svd
import copy
import cv2
from cv2 import dct, idct
from pywt import dwt2, idwt2
from .pool import AutoPool
from .util import *
from .logging import Log
from .base import Steganorgraphy

class DCT_core:
    def __init__(self, password_img=1, mode='common', processes=None):
        self.block_shape = np.array([4, 4])
        self.password_img = password_img
        self.d1, self.d2 = 36, 20
        self.log = Log('DCT')
        self.log.info(f"Running process is {mode}, processes is {processes}")
        # init data
        self.img, self.img_YUV = None, None
        self.ca, self.hvd, = [np.array([])] * 3, [np.array([])] * 3 
        self.ca_block = [np.array([])] * 3 
        self.ca_part = [np.array([])] * 3 

        self.wm_size, self.block_num = 0, 0
        self.pool = AutoPool(mode=mode, processes=processes)

        self.fast_mode = False
        self.alpha = None 

    def init_block_index(self):
        self.block_num = self.ca_block_shape[0] * self.ca_block_shape[1]
        if self.wm_size > self.block_num:
            self.log.warning('The maximum size of watermark is {} kb, flow out {} kb'.format(self.block_num / 1000, self.wm_size / 1000))
        assert self.wm_size < self.block_num, IndexError('The maximum size of watermark is {} kb, flow out {} kb'.format(self.block_num / 1000, self.wm_size / 1000))
        self.part_shape = self.ca_block_shape[:2] * self.block_shape
        self.block_index = [(i, j) for i in range(self.ca_block_shape[0]) for j in range(self.ca_block_shape[1])]

    def read_img_arr(self, img):
        self.alpha = None
        if img.shape[2] == 4:
            if img[:, :, 3].min() < 255:
                self.alpha = img[:, :, 3]
                img = img[:, :, :3]

        self.img = img.astype(np.float32)
        self.img_shape = self.img.shape[:2]

        self.img_YUV = cv2.copyMakeBorder(cv2.cvtColor(self.img, cv2.COLOR_BGR2YUV),
                                          0, self.img.shape[0] % 2, 0, self.img.shape[1] % 2,
                                          cv2.BORDER_CONSTANT, value=(0, 0, 0))

        self.ca_shape = [(i + 1) // 2 for i in self.img_shape]

        self.ca_block_shape = (self.ca_shape[0] // self.block_shape[0], self.ca_shape[1] // self.block_shape[1],
                               self.block_shape[0], self.block_shape[1])
        strides = 4 * np.array([self.ca_shape[1] * self.block_shape[0], self.block_shape[1], self.ca_shape[1], 1])

        for channel in range(3):
            self.ca[channel], self.hvd[channel] = dwt2(self.img_YUV[:, :, channel], 'haar')
            self.ca_block[channel] = np.lib.stride_tricks.as_strided(self.ca[channel].astype(np.float32),
                                                                     self.ca_block_shape, strides)

    def read_wm(self, wm_bit):
        self.wm_bit = wm_bit
        self.wm_size = wm_bit.size

    def block_add_wm(self, arg):
        if self.fast_mode:
            return self.block_add_wm_fast(arg)
        else:
            return self.block_add_wm_slow(arg)

    def block_add_wm_slow(self, arg):
        block, shuffler, i = arg
        wm_1 = self.wm_bit[i % self.wm_size]
        block_dct = dct(block)

        block_dct_shuffled = block_dct.flatten()[shuffler].reshape(self.block_shape)
        u, s, v = svd(block_dct_shuffled)
        s[0] = (s[0] // self.d1 + 1 / 4 + 1 / 2 * wm_1) * self.d1
        if self.d2:
            s[1] = (s[1] // self.d2 + 1 / 4 + 1 / 2 * wm_1) * self.d2

        block_dct_flatten = np.dot(u, np.dot(np.diag(s), v)).flatten()
        block_dct_flatten[shuffler] = block_dct_flatten.copy()
        return idct(block_dct_flatten.reshape(self.block_shape))

    def block_add_wm_fast(self, arg):
        block, shuffler, i = arg
        wm_1 = self.wm_bit[i % self.wm_size]

        u, s, v = svd(dct(block))
        s[0] = (s[0] // self.d1 + 1 / 4 + 1 / 2 * wm_1) * self.d1

        return idct(np.dot(u, np.dot(np.diag(s), v)))

    def embed(self):
        self.init_block_index()

        embed_ca = copy.deepcopy(self.ca)
        embed_YUV = [np.array([])] * 3

        self.idx_shuffle = random_strategy1(self.password_img, self.block_num,
                                            self.block_shape[0] * self.block_shape[1])
        for channel in range(3):
            tmp = self.pool.map(self.block_add_wm,
                                [(self.ca_block[channel][self.block_index[i]], self.idx_shuffle[i], i)
                                 for i in range(self.block_num)])

            for i in range(self.block_num):
                self.ca_block[channel][self.block_index[i]] = tmp[i]

            self.ca_part[channel] = np.concatenate(np.concatenate(self.ca_block[channel], 1), 1)
            embed_ca[channel][:self.part_shape[0], :self.part_shape[1]] = self.ca_part[channel]
            embed_YUV[channel] = idwt2((embed_ca[channel], self.hvd[channel]), "haar")

        embed_img_YUV = np.stack(embed_YUV, axis=2)
        embed_img_YUV = embed_img_YUV[:self.img_shape[0], :self.img_shape[1]]
        embed_img = cv2.cvtColor(embed_img_YUV, cv2.COLOR_YUV2BGR)
        embed_img = np.clip(embed_img, a_min=0, a_max=255)

        if self.alpha is not None:
            embed_img = cv2.merge([embed_img.astype(np.uint8), self.alpha])
        self.log.info(f"watermark is {self.wm_size} bits")
        return embed_img

    def block_get_wm(self, args):
        if self.fast_mode:
            return self.block_get_wm_fast(args)
        else:
            return self.block_get_wm_slow(args)

    def block_get_wm_slow(self, args):
        block, shuffler = args
        block_dct_shuffled = dct(block).flatten()[shuffler].reshape(self.block_shape)

        u, s, v = svd(block_dct_shuffled)
        wm = (s[0] % self.d1 > self.d1 / 2) * 1
        if self.d2:
            tmp = (s[1] % self.d2 > self.d2 / 2) * 1
            wm = (wm * 3 + tmp * 1) / 4
        return wm

    def block_get_wm_fast(self, args):
        block, shuffler = args
        u, s, v = svd(dct(block))
        wm = (s[0] % self.d1 > self.d1 / 2) * 1

        return wm

    def extract_raw(self, img):
        self.read_img_arr(img=img)
        self.init_block_index()

        wm_block_bit = np.zeros(shape=(3, self.block_num))  # 3个channel，length 个分块提取的水印，全都记录下来

        self.idx_shuffle = random_strategy1(seed=self.password_img,
                                            size=self.block_num,
                                            block_shape=self.block_shape[0] * self.block_shape[1],  # 16
                                            )
        for channel in range(3):
            wm_block_bit[channel, :] = self.pool.map(self.block_get_wm,
                                                     [(self.ca_block[channel][self.block_index[i]], self.idx_shuffle[i])
                                                      for i in range(self.block_num)])
        return wm_block_bit

    def extract_avg(self, wm_block_bit):
        wm_avg = np.zeros(shape=self.wm_size)
        for i in range(self.wm_size):
            wm_avg[i] = wm_block_bit[:, i::self.wm_size].mean()
        return wm_avg

    def extract(self, img, wm_shape):
        self.wm_size = np.array(wm_shape).prod()

        wm_block_bit = self.extract_raw(img=img)
        wm_avg = self.extract_avg(wm_block_bit)
        return wm_avg

    def extract_with_kmeans(self, img, wm_shape):
        wm_avg = self.extract(img=img, wm_shape=wm_shape)

        return one_dim_kmeans(wm_avg)


def one_dim_kmeans(inputs):
    threshold = 0
    e_tol = 10 ** (-6)
    center = [inputs.min(), inputs.max()] 
    for i in range(300):
        threshold = (center[0] + center[1]) / 2
        is_class01 = inputs > threshold 
        center = [inputs[~is_class01].mean(), inputs[is_class01].mean()] 
        if np.abs((center[0] + center[1]) / 2 - threshold) < e_tol:
            threshold = (center[0] + center[1]) / 2
            break

    is_class01 = inputs > threshold
    return is_class01


def random_strategy1(seed, size, block_shape):
    return np.random.RandomState(seed) \
        .random(size=(size, block_shape)) \
        .argsort(axis=1)


def random_strategy2(seed, size, block_shape):
    one_line = np.random.RandomState(seed) \
        .random(size=(1, block_shape)) \
        .argsort(axis=1)

    return np.repeat(one_line, repeats=size, axis=0)

def encode(string):
    byte = bin(int(string.encode('utf-8').hex(), base=16))[2:]
    return (np.array(list(byte)) == '1')

class DCT(Steganorgraphy):
    '''
    The DCT algorithm is used to embed the watermark into the image. It divides the image into 8x8 blocks, and then uses the DCT to transform the blocks.
    The DCT coefficients are then shuffled, and the watermark is embedded into the shuffled coefficients. 
    The watermark is then extracted from the shuffled coefficients.

    ## examples::
    ### embeding
    >>> encoder = DCT(password_img=1, password_wm=1, mode='common', processes=None)
    >>> file = File('your_path_to_image.png')
    >>> wm = "hellow world"
    >>> encoder.embed(file=file, wm=wm, mode='text', out_name='output.png')

    ### extracting
    >>> decoder = DCT(password_img=1, password_wm=1, mode='common', processes=None)
    >>> file = File('output.png')
    >>> wm_shape = 87 # the length of watermark's bits
    >>> wm = decoder.extract(embed_img="test.png", wm_shape=32, mode='text')
    >>> print(wm) # hellow world
    '''
    def __init__(self, password_wm=1, password_img=1, mode='common', processes=None):
        '''
        :params password_wm: the seed that use to encrypt the watermark
        :params password_img: the seed that use to encrypt the image
        :params mode: the mode of the algorithm, 'common', 'multiprocessing', 'multithreading'
        :params processes: the num of the process
        '''
        super(DCT).__init__()
        self.bwm_core = DCT_core(password_img=password_img, mode=mode, processes=processes)
        self.password_wm = password_wm
        self.wm_bit = None
        self.wm_size = 0
    
    def load(self, file=None):
        assert file is not None , "file must be not None"
        byteio = file.bdata.hex
        _ = np.frombuffer(byteio, dtype=np.uint8)
        img = cv2.imdecode(_, cv2.IMREAD_COLOR)
        self.bwm_core.read_img_arr(img=img)

    def __dct_wm(self, content, mode):
        if mode == 'img':
            _ = np.frombuffer(hex2bin(content.bdata.hex), np.uint8)
            wm = cv2.imdecode(_, cv2.IMREAD_GRAYSCALE)
            # wm = cv2.imread(filename=content, flags=cv2.IMREAD_GRAYSCALE)
            assert wm is not None, 'file "{filename}" not read'.format(filename=content)
            self.wm_bit = wm.flatten() > 128
        elif mode == 'text':
            byte = bin(int(content.encode('utf-8').hex(), base=16))[2:]
            self.wm_bit = (np.array(list(byte)) == '1')
        self.wm_size = self.wm_bit.size
        np.random.RandomState(self.password_wm).shuffle(self.wm_bit)
        self.bwm_core.read_wm(self.wm_bit)

    def read_wm(self, wm_content, mode='img'):
        assert mode in ('img', 'text'), "mode in ('img','text')"
        self.__dct_wm(wm_content, mode)
        return self
    
    def embed(self, file=None, wm=None, mode='img', out_name=None):
        '''
        embed the hidden information to the target image

        :params file: the target File object
        :params wm: the watermark File object or the string
        :params mode: the mode of the watermark, 'img' or 'text'
        :params out_name: the output name of the embed image
        :return: the embed image if out_name is None, otherwise return None
        '''
        assert file is not None, "file must be not None"
        self.load(file=file)
        self.read_wm(wm, mode)
        self.log.info(f'Embed {self.wm_size} bits into image size: {self.bwm_core.img_shape}')
        embed_img = self.bwm_core.embed()
        cv2.imwrite(filename=out_name, img=embed_img)
        return embed_img
    
    def extract_decrypt(self, wm_avg):
        wm_index = np.arange(self.wm_size)
        np.random.RandomState(self.password_wm).shuffle(wm_index)
        wm_avg[wm_index] = wm_avg.copy()
        return wm_avg

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
        assert wm_shape is not None, 'wm_shape needed'

        if filename is not None:
            embed_img = cv2.imread(filename, flags=cv2.IMREAD_COLOR)
            assert embed_img is not None, "{filename} not read".format(filename=filename)
        else:
            byteio = embed_img.bdata.hex
            _ = np.frombuffer(byteio, dtype=np.uint8)
            embed_img = cv2.imdecode(_, cv2.IMREAD_COLOR)

        self.wm_size = np.array(wm_shape).prod()
        self.log.info(message=f'Extract {wm_shape} bits from image size: {embed_img.shape[:2]}')
        if mode in ('text', 'bit'):
            wm_avg = self.bwm_core.extract_with_kmeans(img=embed_img, wm_shape=wm_shape)
        else:
            wm_avg = self.bwm_core.extract(img=embed_img, wm_shape=wm_shape)
        wm = self.extract_decrypt(wm_avg=wm_avg)
        if mode == 'img':
            wm = 255 * wm.reshape(wm_shape[0], wm_shape[1])
            self.log.info(message=f'The watermark extracted is: {out_wm_name}')
            cv2.imwrite(out_wm_name, wm)
        elif mode == 'text':
            byte = ''.join(str((i >= 0.5) * 1) for i in wm)
            wm = bytes.fromhex(hex(int(byte, base=2))[2:]).decode('utf-8', errors='replace')
            self.log.info(message=f'The watermark extracted is: {wm}')
        return wm
            

