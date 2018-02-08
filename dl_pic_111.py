import urllib.request
from PIL import Image
import matplotlib.pyplot as plt
import shutil
import time


def download_pic():
    url = 'http://passport.111.com.cn/sso/getSecurityCode.action'

    f = urllib.request.urlopen(url)
    data = f.read()
    with open("d:/captcha/tmp.jpg", "wb") as code:
        code.write(data)

    plt.ion()
    img = Image.open("d:/captcha/tmp.jpg")
    plt.imshow(img)
    plt.pause(1)

    name = input('File name：')
    file_name = "./image_111/" + name + ".jpg"
    shutil.move("tmp.jpg", file_name)

    plt.close()


if __name__ == '__main__':
    while 1:
        download_pic()
