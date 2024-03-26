"""
# Witchy

This library that can allow you to access the binary data of the file and modify the file as you like.
It also provide some method to convert the file into another format.

## Usage
>>> from witchy import File
>>> f = File("test.exe")
"""
from .base import File
from .base import Image_compressor