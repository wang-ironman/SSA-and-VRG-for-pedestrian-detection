# -*- coding: utf-8 -*-
# @Time    : 18-12-10 下午7:42
# @Author  : unicoe
# @Email   : unicoe@163.com
# @File    : mkdir.py
# @Software: PyCharm
def mkdir(path):
    import os

    path = path.strip()
    path = path.rstrip("\\")

    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + 'create done.')
        return True
    else:
        print(path + 'path already exists.')
        return False