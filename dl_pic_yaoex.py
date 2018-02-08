#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/2/01 12:09
# @Author  : caozhiye

import urllib.request
from PIL import Image
import matplotlib.pyplot as plt
import shutil


def download_pic():
    """
    下载验证码并保存成文件
    :return:
    """
    url = 'https://web-ycaptcha.yaoex.com/getcode?glAppId=1016'

    f = urllib.request.urlopen(url)
    data = f.read()
    with open("d:/captcha/tmp.jpg", "wb") as code:
        code.write(data)

    plt.ion()
    img = Image.open("d:/captcha/tmp.jpg")
    plt.imshow(img)
    plt.pause(1)

    name = input('File name：')
    file_name = "./image/" + name + ".jpg"
    shutil.move("tmp.jpg", file_name)
    plt.close()


if __name__ == '__main__':
    while 1:
        download_pic()
