import io
import os

def decode(s:bytes)->str:
    '''
    Decode the bytes into string.
    :params s: the bytes to be decoded.
    :return: the decoded string.
    '''
    return ''.join(chr(int(s[i*8:i*8+8],2)) for i in range(len(s)//8))

def encode(s):
    '''
    Encode the string into bytes.
    :params s: the string to be encoded.
    :return: the encoded bytes.
    '''
    return ''.join(bin(ord(c)).replace('0b','').rjust(8,'0') for c in s)

def hex2bin(hex_str:str)->io.BytesIO:
    byte_stream = io.BytesIO(hex_str)
    return byte_stream