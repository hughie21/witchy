from collections.abc import Iterable
import os
from typing import Any
import time
from .magic import magic
from win32file import CreateFile, SetFileTime, GetFileTime, CloseHandle
from win32file import GENERIC_READ, GENERIC_WRITE, OPEN_EXISTING
from pywintypes import Time
from .pic import Image_Convert, Image
from .doc import PDF_Convert, BytesIO
from .error import *

ERROR_MSG = ""

IMAGE_TYPE = ["PNG", "JPG", "JPEG", "BMP", "BMP_16", "BMP_24", "BMP_256", "GIF"]

DOC_TYPE = ["DOC", "DOCX", "TXT", "PDF"]

VIDEO_TYPE = ["MP4", "AVI", "MOV", "FLV", "WMV", "MKV", "WEBM"]

AUDIO_TYPE = ["MP3", "WAV"]

def tips(args:str, sets:list)->str:
    """
    Remind the user of the possible arguement

    :params args: the wrong args
    :params sets: all the args list
    :return: the possible args for the wrong one
    """
    args = set(args)
    temp = []
    for k in sets:
        temp.append(len([val for val in args if val in k]))
    if max(temp) < 3: 
        return 0
    return sets[temp.index(max(temp))]

class Hex:
    """
    The class that storge the binary data of the file

    :params hex: the binary data
    """
    def __init__(self, hex:bytes=None) -> None:
        self.hex = hex
    
    def __getitem__(self, key:int)->int:
        '''
        Get the part of binary data based on the slice

        :params key: the slice of the data
        '''
        return self.hex[key]

    def __setitem__(self, key:int, value:int)->None:
        '''
        Changing values at a specific location

        :params key: the slice of the data
        '''
        self.hex[key] = value
    
    def __len__(self)->int:
        '''
        Get the length of the binary data
        
        :return: length
        '''
        return len(self.hex)
    
    def __iter__(self)->Iterable[int]:
        '''
        return the iterable object of the class
        '''
        for i in range(len(self.hex)):
            yield self.hex[i]
        
    def __hash__(self) -> int:
        '''
        return the hash of the data
        '''
        return hash(self.hex)

    def __repr__(self)->str:
        return f"Hex(length: {len(self.hex)} hash: {hash(self.hex)})"
    
    def __str__(self) -> str:
        return f"Hex(length: {len(self.hex)} hash: {hash(self.hex)})"

class File:
    """
    # Witchy File

    this class is the major function of the library which can allow you to modify the file's attribute or content as you like.

    ## example::
    ### basic function
    >>> f = File("test.jpg")
    >>> print(f) # show the detail information of the file
    >>> print(f("size")) # get the attribute of the file
    >>> f["ctime"] = "2024-01-01 00:00:00" # change the attribute
    >>> f.append(b"test") # append the binary data on the tail
    >>> f.save("test-1.jpg") # save as another file

    ### convert function
    >>> f = File("test.jpg")
    >>> f.convert("test.png", "PNG") # convert to another format
    """
    def __init__(self, path:str=None) -> None:
        self.info = {}
        if path != None:
            if os.path.exists(path) == False:
                raise FileNotFoundError("File not found")
            self.open(path)
    
    def __merge(self, file):
        if len(file) == 0:
            raise Exception("No images to merge")
        binset = [BytesIO(self.bdata.hex)]
        for f in file:
            if f.info["type"] not in IMAGE_TYPE:
                raise TypeError(f"{f.path} is not a image file")
            binset.append(BytesIO(f.bdata.hex))
        doc = PDF_Convert.merge_pic(binset)
        return doc
    
    def __image_convert(self, to:str, format:str, quality:int, size:tuple, file)->None:
        if format == "JPEG" or format == "JPG":
            cimage = Image_Convert.convert_to_damaged_images(self.bdata.hex, quality)
            cimage.save(to, format="jpg")
        elif format == "PNG":
            cimage = Image_Convert.convert_to_lossless_images(self.bdata.hex)
            cimage.save(to, format="png")
        elif format == "BMP":
            cimage = Image_Convert.convert_to_lossless_images(self.bdata.hex)
            cimage.save(to, format="bmp")
        elif format == "GIF":
            cimage = Image_Convert.convert_to_lossless_images(self.bdata.hex)
            cimage.save(to, format="gif")
        elif format == "ICO":
            icon_sizes = [(16, 16), (32, 32), (48, 48), (64, 64)]
            if size not in icon_sizes:
                raise ValueError(f"Invalid icon size. the image size must include {icon_sizes}")
            cimage = Image_Convert.convert_to_lossless_images(self.bdata.hex)
            cimage.save(to, format="ico", size=size)
        elif format == "PDF":
            doc = self.__merge(file)
            doc.save(to)

    def __doc_convert(self, to:str, format:str)->None:
        if format == "PIC":
            image_data = PDF_Convert.to_image(self.bdata.hex)
            name,suffix = os.path.splitext(to)
            for k,v in image_data.items():
                v.save(f"{name}-{k}.{suffix}")
        elif format == "DOC":
            word = PDF_Convert.to_doc(self.bdata.hex)
            word.save(to)
        elif format == "TXT":
            text = PDF_Convert.to_text(self.bdata.hex)
            with open(f"{to}", "w+", encoding="utf-8") as f:
                f.write(text)
                

    def convert(self, to:str, format:str, quality:int = 100, size=(64,64), file=())->str:
        format = format.upper()
        if self.info["type"] in IMAGE_TYPE:
            self.__image_convert(to, format, quality, size, file)
        elif self.info["type"] in DOC_TYPE:
            self.__doc_convert(to, format)

    def __checkMagic(self):
        b:str = self.bdata[:28].hex().upper()
        for k,v in magic.items():
            for i in range(0, len(b)-len(v)):
                if b[i:i+len(v)] == v:
                    return k
        else:
            return "Unknown"
    
    def append(self, data:Any)->None:
        if isinstance(data, bytes):
            self.bdata.hex = self.bdata.hex + data
        elif isinstance(data, Hex):
            self.bdata.hex = self.bdata.hex + data.hex
        else:
            self.bdata.hex = self.bdata.hex + bytes(data, "UTF-8")
    
    def save(self, path:str)->None:
        with open(path, "wb") as f:
            f.write(self.bdata.hex)
        fh = CreateFile(path, GENERIC_READ | GENERIC_WRITE, 0, None, OPEN_EXISTING, 0, 0)
        createTimes, accessTimes, modifyTimes = GetFileTime(fh)
        createTimes = Time(self.info["ctime"])
        accessTimes = Time(self.info["atime"])
        modifyTimes = Time(self.info["mtime"])
        SetFileTime(fh, createTimes, accessTimes, modifyTimes)
        CloseHandle(fh)

    def open(self,path:str)->None:
        if os.path.exists(path) == False:
            raise FileNotFoundError("File not found")
        self.path = path
        stat = os.stat(path)
        self.info = {
            "st_uid": stat.st_uid,
            "st_gid": stat.st_gid,
            "size": stat.st_size,
            "ctime": stat.st_ctime,
            "atime": stat.st_atime,
            "mtime": stat.st_mtime,
        }
        try:
            f = open(path, "rb")
            self.bdata = Hex(f.read())
        except Exception:
            raise Exception("failed to open file")
        finally:
            f.close()
        self.info["type"] = self.__checkMagic()

    def __setitem__(self, __name: str, __value: Any) -> None:
        timestamp = time.mktime(time.strptime(__value, "%Y-%m-%d %H:%M:%S"))
        if timestamp > 2147483647.0:
            raise TimeOutRangeException("the maximum of timestamp is 2147483647 which is 2038")
        if __name == "atime":
            self.info["atime"] = timestamp
        elif __name == "mtime":
            self.info["mtime"] = timestamp
        elif __name == "ctime":
            self.info["ctime"] = timestamp
        else:
            possible = tips(__name, ["ctime","atime", "mtime"])
            if possible == 0:
                raise KeyErrorException(f"invalid argument '{__name}', pleace check the document")
            raise KeyErrorException(f"invalid argument '{__name}'. Do you mean '{possible}'?")

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        args = args[0]
        if args == "uid":
            return self.info["st_uid"]
        elif args == "gid":
            return self.info["st_gid"]
        elif args == "size":
            return self.info["size"]
        elif args == "atime":
            return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.info["atime"]))
        elif args == "mtime":
            return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.info["mtime"]))
        elif args == "ctime":
            return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.info["ctime"]))
        elif args == "data":
            return self.bdata
        elif args == "path":
            return os.path.abspath(self.path)
        else:
            possible = tips(args,["uid", "gid", "size", "atime", "ctime", "mtime", "data", "path"])
            if possible == 0:
                ERROR_MSG = f"invalid argument '{args}', pleace check the document"
                raise KeyErrorException(ERROR_MSG)
            ERROR_MSG = f"invalid argument '{args}'. Do you mean '{possible}'?"
            raise KeyErrorException(ERROR_MSG)
    
    def __str__(self) -> str:
        return f"File:(path: {os.path.abspath(self.path)}, type: {self.info['type']}, uid: {self.info['st_uid']}, gid: {self.info['st_gid']}, size: {self.info['size']}, atime: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.info['atime']))}, mtime: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.info['mtime']))},  ctime: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.info['ctime']))}, bin_data: {str(self.bdata)})"

    def __repr__(self) -> str:
        return f"File:(path: {os.path.abspath(self.path)}, type: {self.info['type']}, uid: {self.info['st_uid']}, gid: {self.info['st_gid']}, size: {self.info['size']}, atime: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.info['atime']))}, mtime: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.info['mtime']))}, ctime: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.info['ctime']))}, bin_data: {str(self.bdata)})"
