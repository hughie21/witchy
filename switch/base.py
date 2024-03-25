from collections.abc import Iterable
import os
from typing import Any
import time
from .magic import magic
from win32file import CreateFile, SetFileTime, GetFileTime, CloseHandle
from win32file import GENERIC_READ, GENERIC_WRITE, OPEN_EXISTING
from pywintypes import Time
from .pic import Image_Convert
from .error import *

ERROR_MSG = ""

IMAGE_TYPE = ["PNG", "JPG", "JPEG", "BMP", "BMP_16", "BMP_24", "BMP_256", "GIF"]

DOC_TYPE = ["DOC", "DOCX", "TXT", "PDF", "XLS", "XLSX", "PPT", "PPTX"]

VIDEO_TYPE = ["MP4", "AVI", "MOV", "FLV", "WMV", "MKV", "WEBM"]

AUDIO_TYPE = ["MP3", "WAV", "FLAC", "AAC", "OGG", "M4A"]

def tips(args:str, sets:list):
    args = set(args)
    temp = []
    for k in sets:
        temp.append(len([val for val in args if val in k]))
    if max(temp) < 3:
        return 0
    return sets[temp.index(max(temp))]

class Hex:
    def __init__(self, hex:bytes=None) -> None:
        self.hex = hex
    def __getitem__(self, key:int)->int:
        return self.hex[key]
    def __setitem__(self, key:int, value:int)->None:
        self.hex[key] = value
    def __len__(self)->int:
        return len(self.hex)
    def __iter__(self)->Iterable[int]:
        for i in range(len(self.hex)):
            yield self.hex[i]
    def __hash__(self) -> int:
        return hash(self.hex)
    def __repr__(self)->str:
        return f"Hex(length: {len(self.hex)} hash: {hash(self.hex)})"
    def __str__(self) -> str:
        return f"Hex(length: {len(self.hex)} hash: {hash(self.hex)})"

class File:
    def __init__(self, path:str=None) -> None:
        self.info = {}
        if path != None:
            if os.path.exists(path) == False:
                raise FileNotFoundError("File not found")
            self.open(path)
    
    def __image_convert(self, to:str, format:str, quality:int, size:tuple)->None:
        if format == "JPEG" or format == "JPG":
            cimage = Image_Convert.convert_to_damaged_images(self.bdata.hex, quality)
            cimage.save(to)
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

    def convert(self, to:str, format:str, quality:int = 100, size=(64,64))->str:
        format = format.upper()
        if self.info["type"] in IMAGE_TYPE:
            self.__image_convert(to, format, quality, size)

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
        if __name == "atime":
            self.info["atime"] = time.mktime(time.strptime(__value, "%Y-%m-%d %H:%M:%S"))
        elif __name == "mtime":
            self.info["mtime"] = time.mktime(time.strptime(__value, "%Y-%m-%d %H:%M:%S"))
        elif __name == "ctime":
            self.info["ctime"] = time.mktime(time.strptime(__value, "%Y-%m-%d %H:%M:%S"))
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
