from ffmpy3 import FFmpeg
import os

class Media_Convert:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def check_environment():
        cmd = "ffmpeg -version"
        res = os.system(cmd)
        if res == 0:
            return True
        return False
    
    @staticmethod
    def to_audio(path:str, to:str, type:str):
        name, suffix = os.path.splitext(to)
        ff =FFmpeg(inputs={path: None}, outputs={f"{name}.{type}": None})
        ff.run()

    @staticmethod
    def to_vedio(path:str, to:str, type:str):
        name, suffix = os.path.splitext(to)
        ff =FFmpeg(inputs={path: None}, outputs={f"{name}.{type}": None})
        ff.run()