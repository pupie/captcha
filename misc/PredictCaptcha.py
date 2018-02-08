#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/2/6 16:46
# @Author  : caozhiye

import tensorflow as tf
import numpy as np
from PIL import Image
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽tensorflow运行时CPU指令集警告
CAPTCHA_LEN = 4

absolute_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../file/b2bpc/captcha_cnn/model")) + "\\"

# MODEL_SAVE_PATH = 'D:/captcha/model/'
MODEL_SAVE_PATH = absolute_path
TEST_IMAGE_PATH = 'C:/'


def get_image_data_and_name(file_name, file_path=TEST_IMAGE_PATH):
    """
    获取图片文件名和扁平化数组
    :param file_name: 文件名如 captcha.jpg
    :param file_path: 文件路径
    :return: 扁平化数组，文件名
    """
    path_name = os.path.join(file_path, file_name)
    img = Image.open(path_name)
    # 转为灰度图
    img = img.convert("L")
    image_array = np.array(img)
    image_data = image_array.flatten() / 255
    image_name = file_name[0:CAPTCHA_LEN]
    return image_data, image_name


def predict_captcha(file_name="captcha.jpg"):
    """
    根据验证码图片预测验证码
    :param file_name: 验证码文件名
    :return: 验证码字符串
    """
    # 加载graph
    saver = tf.train.import_meta_graph(MODEL_SAVE_PATH + "crack_captcha.model.meta")
    graph = tf.get_default_graph()

    # 从graph取得tensor，他们的name是在构建graph时定义的
    input_holder = graph.get_tensor_by_name("data-input:0")
    keep_prob_holder = graph.get_tensor_by_name("keep-prob:0")
    predict_max_idx = graph.get_tensor_by_name("predict_max_idx:0")

    #  预测验证码
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(MODEL_SAVE_PATH))
        img_data, img_name = get_image_data_and_name(file_name, TEST_IMAGE_PATH)
        predict = sess.run(predict_max_idx, feed_dict={input_holder: [img_data], keep_prob_holder: 1.0})
        file_path_name = TEST_IMAGE_PATH + file_name  # 计算文件绝对路径
        predict_value = np.squeeze(predict).tolist()  # 预测验证码并将结果numpy.ndarray 转化成list

        # print(predict_value)  # [3, 4, 2, 5]，整数为元素的列表
        # 将列表元素转换成字符串拼接输出
        captcha_text = ""
        for item in predict_value:
            text = str(item)
            captcha_text = captcha_text + text

        print("file name:%s predicted captcha:%s" % (file_path_name, captcha_text))
        return captcha_text


if __name__ == '__main__':
    predict_captcha()
