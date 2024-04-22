from collections.abc import Iterable
import os
from typing import Any
import time
from .magic import magic
from win32file import CreateFile, SetFileTime, GetFileTime, CloseHandle
from win32file import GENERIC_READ, GENERIC_WRITE, OPEN_EXISTING
from pywintypes import Time
from .error import *
from hashlib import md5

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
    >>> f = File("path/your/file")
    >>> print(f) # show the detail information of the file
    >>> print(f("size")) # get the attribute of the file
    >>> f["ctime"] = "2024-01-01 00:00:00" # change the attribute
    >>> f.append(b"your data") # append the binary data on the tail
    >>> f.save("path/tour/file") # save as another file

    ### convert function
    >>> f = File("your.jpg")
    >>> f.convert("output.png", "PNG") # convert to another format
    """
    def __init__(self, path:str=None) -> None:
        self.info = {}
        if path != None:
            if os.path.exists(path) == False:
                raise FileNotFoundError("File not found")
            self.open(path)

    def find(self, target:str|bytes):
        pass

    def __checkMagic(self)->str:
        '''
        based on the magic number to identify the type of the file

        :return: file type
        '''
        b:str = self.bdata[:28].hex().upper()
        for k,v in magic.items():
            if v in b:
                return k
        else:
            return "Unknown"
    
    def append(self, data:Any)->None:
        '''
        append the binary data on the tail of the file

        :params data: the data that could be File class or sting like type
        '''
        if isinstance(data, bytes):
            self.bdata.hex = self.bdata.hex + data
        elif isinstance(data, Hex):
            self.bdata.hex = self.bdata.hex + data.hex
        else:
            self.bdata.hex = self.bdata.hex + bytes(data, "UTF-8")
    
    def save(self, path:str)->None:
        '''
        save as another file

        :params path: the output path
        '''
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
        elif args == "type":
            return self.info["type"]
        else:
            possible = tips(args,["uid", "gid", "size", "atime", "ctime", "mtime", "data", "path", "type"])
            if possible == 0:
                ERROR_MSG = f"invalid argument '{args}', pleace check the document"
                raise KeyErrorException(ERROR_MSG)
            ERROR_MSG = f"invalid argument '{args}'. Do you mean '{possible}'?"
            raise KeyErrorException(ERROR_MSG)
    
    def __eq__(self, value: object) -> bool:
        if md5(self.bdata.hex) == md5(value.bdata.hex):
            return True
        else:
            return False

    def __str__(self) -> str:
        return f"File:(path: {os.path.abspath(self.path)}, type: {self.info['type']}, uid: {self.info['st_uid']}, gid: {self.info['st_gid']}, size: {self.info['size']}, atime: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.info['atime']))}, mtime: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.info['mtime']))},  ctime: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.info['ctime']))}, bin_data: {str(self.bdata)})"

    def __repr__(self) -> str:
        return f"File:(path: {os.path.abspath(self.path)}, type: {self.info['type']}, uid: {self.info['st_uid']}, gid: {self.info['st_gid']}, size: {self.info['size']}, atime: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.info['atime']))}, mtime: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.info['mtime']))}, ctime: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.info['ctime']))}, bin_data: {str(self.bdata)})"
