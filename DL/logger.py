import os, sys
import logging
import datetime
from logging import handlers
class logger:
    def __init__(self, base_filename):
        base_name, base_ext = base_filename.split('.')[0], base_filename.split('.')[1]
        filename = base_name+ '_' + datetime.datetime.today().strftime("%Y%m%d%H%M") + '.' + base_ext

        self.log = logging.getLogger(filename)
        sh = logging.StreamHandler()
        fh = handlers.TimedRotatingFileHandler()
        logging.add
        print(filename)

