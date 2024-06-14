import shutil
import time
from datetime import datetime
import const.ds_const as CONST
import os

class DLlogger(object):
    """Логгер для вывода в файл"""
    def __init__(self, fn='', tofile=False):
	    self.fn = fn
	    self.tofile = tofile
	    return
    def printml(self, *args):
        toprint = ''
        for v in args:
            toprint = toprint + str(v) + ' '
        if self.tofile:
            f = open(self.fn, 'a')
            f.write(toprint + "\n")
            f.close()
        else: print(toprint)
        return



def time_logger(func):
    """Простой декоратор, нужен только для логирования времени."""

    def wrap(*args, **kwargs):
        start = time.time()
        print(f'Выполнение функции "{CONST.func_description[func.__name__]}" начало:{datetime.now()}')
        result = func(*args, **kwargs)
        end = time.time()
        print(f'Выполнение функции "{CONST.func_description[func.__name__]}" завершение: {datetime.now()}')
        print(f'Выполнение функции "{CONST.func_description[func.__name__]}" длительность: {end - start}')
        return result

    return wrap

def create_rep_dir(target):
    if not os.path.exists(f'report/{target}'):
        os.makedirs(f'report/{target}')

def move_files(filename, target):
    shutil.move(filename, f'report/{target}/img/{filename}')
