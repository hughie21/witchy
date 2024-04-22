import logging
import logging.config
from .config import *

if not os.path.exists(LOGGING_PATH):
    os.makedirs(LOGGING_PATH)

if not os.path.exists(LOGGING_FILENAME):
    with open(LOGGING_FILENAME, 'w+') as f:
        f.write('')

class Log:
    def __init__(self, mode):
        self.logger = logging.getLogger('logger')
        if mode is not None:
            formats= LOGGING_FORMAT.format(mode)
        else:
            formats = LOGGING_FORMAT.format('Unknown')
        handler1 = logging.StreamHandler()
        handler2 = logging.FileHandler(filename=LOGGING_FILENAME, mode='a')
        self.logger.setLevel(logging.INFO)
        handler1.setLevel(logging.INFO)
        handler2.setLevel(logging.INFO)
        formatter = logging.Formatter(formats)
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        self.logger.addHandler(handler1)
        self.logger.addHandler(handler2)    

    def error(self, message):
        self.logger.error(message)

    def info(self, message):
        self.logger.info(message)
    
    def warning(self, message):
        self.logger.critical(message)