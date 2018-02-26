#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/2/06 11:00
# @Author  : caozhiye

import tensorflow as tf
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

# 本工具通过卷积神经网络计算验证码模型准确度
absolute_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "./model_111")) + "\\"
absolute_image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "./test_111")) + "\\"
model_file_name = "crack_captcha.model-2600.meta"

# 验证码字符串长度
CAPTCHA_LEN = 4

# 存放训练好的模型的路径
MODEL_SAVE_PATH = absolute_model_path

# 存放用于验证的验证码图片的路径
TEST_IMAGE_PATH = absolute_image_path


def get_image_data_and_name(file_name, file_path=TEST_IMAGE_PATH):
    path_name = os.path.join(file_path, file_name)
    img = Image.open(path_name)
    # 转为灰度图
    img = img.convert("L")
    image_array = np.array(img)
    image_data = image_array.flatten() / 255  # 图像数组扁平化
    image_name = file_name[0:CAPTCHA_LEN]
    return image_data, image_name


def digital_str2_list(digital_str):  # 文件名转成列表
    digital_list = []
    for c in digital_str:
        digital_list.append(c)
    return digital_list


def model_test():
    name_list = []
    for path_name in os.listdir(TEST_IMAGE_PATH):
        name_list.append(path_name.split('/')[-1])
    total_number = len(name_list)

    # 加载graph
    saver = tf.train.import_meta_graph(MODEL_SAVE_PATH + model_file_name)
    graph = tf.get_default_graph()

    # 从graph取得 tensor，他们的name是在构建graph时定义的
    input_holder = graph.get_tensor_by_name("data-input:0")
    keep_prob_holder = graph.get_tensor_by_name("keep-prob:0")
    predict_max_idx = graph.get_tensor_by_name("predict_max_idx:0")
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(MODEL_SAVE_PATH))
        count = 0
        for fileName in name_list:
            img_data, img_name = get_image_data_and_name(fileName, TEST_IMAGE_PATH)
            predict = sess.run(predict_max_idx, feed_dict={input_holder: [img_data], keep_prob_holder: 1.0})
            file_path_name = TEST_IMAGE_PATH + fileName
            print(file_path_name)
            img = Image.open(file_path_name)
            plt.imshow(img)
            plt.axis('off')
            # plt.show() # 屏蔽显示，需要查看图形的时候打开
            predict_value = np.squeeze(predict).tolist()  # 预测结果转np数组转换成列表
            right_value = digital_str2_list(img_name)

            for i, c in enumerate(predict_value):  # 枚举并转换列表元素从10进制数值成16进制字符串
                predict_value[i] = str(hex(c))[-1]

            if predict_value == right_value:
                result = '正确'
                count += 1
            else:
                result = '错误'
                # plt.show()
            print('实际值：{}， 预测值：{}，测试结果：{}'.format(right_value, predict_value, result))
            print('\n')

        # 打印正确率，2018年2月8日，1600次循环计算结果为99.40%
        print('正确率：%.2f%%(%d/%d)' % (count * 100 / total_number, count, total_number))


if __name__ == '__main__':
    model_test()
