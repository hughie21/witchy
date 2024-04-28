import os
import time

EXCPATH = os.getcwd()

LOGGING_LEVEL = 'INFO'

LOGGING_FORMAT = 'Module({}) %(asctime)s : [%(levelname)s] -> %(message)s'

LOGGING_DATE_FORMAT = '%Y-%m-%d %H%M'

LOGGING_PATH = os.path.join(EXCPATH, 'logs')

TIME_NOW = time.strftime(LOGGING_DATE_FORMAT, time.localtime())

LOGGING_FILENAME = os.path.join(LOGGING_PATH ,f'{TIME_NOW}.log')