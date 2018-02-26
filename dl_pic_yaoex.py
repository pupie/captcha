#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/2/01 12:09
# @Author  : caozhiye

import urllib.request
from PIL import Image
import matplotlib.pyplot as plt
import shutil
import os

absolute_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "./tmp.jpg"))


def download_pic():
    """
    下载验证码并保存成文件
    :return:
    """
    url = 'https://web-ycaptcha.yaoex.com/getcode?glAppId=1016'

    f = urllib.request.urlopen(url)
    data = f.read()
    with open(absolute_path, "wb") as code:
        code.write(data)

    plt.ion()
    img = Image.open(absolute_path)
    plt.imshow(img)
    plt.pause(1)

    name = input('File name：')
    file_name = "./image_yaoex/" + name + ".jpg"
    shutil.move("tmp.jpg", file_name)
    plt.close()


if __name__ == '__main__':
    while 1:
        download_pic()
