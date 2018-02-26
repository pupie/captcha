#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/26 13:00
# @Author  : caozhiye

import tensorflow as tf
import numpy as np
from PIL import Image
import os
import random
import time
import datetime

# 本工具通过卷积神经网络深度学习1号药城验证码并保存模型
absolute_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "./model_111")) + "\\"
absolute_image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "./image_111")) + "\\"
model_file_name = "crack_captcha.model"

# 验证码图片的存放路径
CAPTCHA_IMAGE_PATH = absolute_image_path

# 验证码图片的宽度
CAPTCHA_IMAGE_WIDTH = 100

# 验证码图片的高度
CAPTCHA_IMAGE_HEIGHT = 40

# 验证码字符集数量（0-9）
CHAR_SET_LEN = 10

# 验证码字符串长度
CAPTCHA_LEN = 4

# 80%的验证码图片放入训练集中
TRAIN_IMAGE_PERCENT = 0.8

# 训练集，用于训练的验证码图片的文件名
TRAINING_IMAGE_NAME = []

# 验证集，用于模型验证的验证码图片的文件名
VALIDATION_IMAGE_NAME = []

# 存放训练好的模型的路径
MODEL_SAVE_PATH = absolute_model_path


def get_image_file_name(img_path=CAPTCHA_IMAGE_PATH):
    """
    获取用于训练的文件名列表和总数量
    :param img_path:
    :return:
    """
    file_name = []
    total_number = 0
    for filePath in os.listdir(img_path):
        captcha_name = filePath.split('/')[-1]
        file_name.append(captcha_name)
        total_number += 1
    return file_name, total_number


# 将验证码转换为训练时用的标签向量，维数是 40
# 例如，如果验证码是 ‘0296’ ，则对应的标签是
# [1 0 0 0 0 0 0 0 0 0  
#  0 0 1 0 0 0 0 0 0 0  
#  0 0 0 0 0 0 0 0 0 1  
#  0 0 0 0 0 0 1 0 0 0]  
def name2label(name):
    label = np.zeros(CAPTCHA_LEN * CHAR_SET_LEN)
    for i, c in enumerate(name):
        idx = i * CHAR_SET_LEN + ord(c) - ord('0')
        label[idx] = 1
    return label


# 取得验证码图片的数据以及它的标签
def get_data_and_label(file_name, file_path=CAPTCHA_IMAGE_PATH):
    path_name = os.path.join(file_path, file_name)
    img = Image.open(path_name)
    # 转为灰度图
    img = img.convert("L")
    image_array = np.array(img)
    image_data = image_array.flatten() / 255
    image_label = name2label(file_name[0:CAPTCHA_LEN])
    return image_data, image_label


# 生成一个训练batch
def get_next_batch(batch_size=32, train_or_test='train', step=0):
    batch_data = np.zeros([batch_size, CAPTCHA_IMAGE_WIDTH * CAPTCHA_IMAGE_HEIGHT])
    batch_label = np.zeros([batch_size, CAPTCHA_LEN * CHAR_SET_LEN])
    file_name_list = TRAINING_IMAGE_NAME
    if train_or_test == 'validate':
        file_name_list = VALIDATION_IMAGE_NAME

    total_number = len(file_name_list)
    index_start = step * batch_size
    for i in range(batch_size):
        index = (i + index_start) % total_number
        name = file_name_list[index]
        img_data, img_label = get_data_and_label(name)
        batch_data[i, :] = img_data
        batch_label[i, :] = img_label

    return batch_data, batch_label


# 构建卷积神经网络(Convolutional Neural Network, CNN)并训练
def train_data_with_cnn():
    # 初始化权值
    def weight_variable(shape, name='weight'):
        init = tf.truncated_normal(shape, stddev=0.1)
        var = tf.Variable(initial_value=init, name=name)
        return var

    # 初始化偏置
    def bias_variable(shape, name='bias'):
        init = tf.constant(0.1, shape=shape)
        var = tf.Variable(init, name=name)
        return var

    # 卷积
    def conv2d(x, W, name='conv2d'):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=name)

    # 池化
    def max_pool_2X2(x, name='maxpool'):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    # 输入层
    # 请注意 X 的 name，在测试model时会用到它
    X = tf.placeholder(tf.float32, [None, CAPTCHA_IMAGE_WIDTH * CAPTCHA_IMAGE_HEIGHT], name='data-input')
    Y = tf.placeholder(tf.float32, [None, CAPTCHA_LEN * CHAR_SET_LEN], name='label-input')
    x_input = tf.reshape(X, [-1, CAPTCHA_IMAGE_HEIGHT, CAPTCHA_IMAGE_WIDTH, 1], name='x-input')

    # dropout，防止过拟合
    # 请注意 keep_prob 的 name，在测试model时会用到它
    keep_prob = tf.placeholder(tf.float32, name='keep-prob')

    # 第一层卷积
    W_conv1 = weight_variable([5, 5, 1, 32], 'W_conv1')
    B_conv1 = bias_variable([32], 'B_conv1')
    conv1 = tf.nn.relu(conv2d(x_input, W_conv1, 'conv1') + B_conv1)
    conv1 = max_pool_2X2(conv1, 'conv1-pool')
    conv1 = tf.nn.dropout(conv1, keep_prob)

    # 第二层卷积
    W_conv2 = weight_variable([5, 5, 32, 64], 'W_conv2')
    B_conv2 = bias_variable([64], 'B_conv2')
    conv2 = tf.nn.relu(conv2d(conv1, W_conv2, 'conv2') + B_conv2)
    conv2 = max_pool_2X2(conv2, 'conv2-pool')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    # 第三层卷积
    W_conv3 = weight_variable([5, 5, 64, 64], 'W_conv3')
    B_conv3 = bias_variable([64], 'B_conv3')
    conv3 = tf.nn.relu(conv2d(conv2, W_conv3, 'conv3') + B_conv3)
    conv3 = max_pool_2X2(conv3, 'conv3-pool')
    conv3 = tf.nn.dropout(conv3, keep_prob)

    # 全链接层
    # 每次池化后，图片的宽度和高度均缩小为原来的一半，进过上面的三次池化，宽度和高度均缩小8倍
    W_fc1 = weight_variable([13 * 5 * 64, 1024], 'W_fc1')
    B_fc1 = bias_variable([1024], 'B_fc1')
    fc1 = tf.reshape(conv3, [-1, 13 * 5 * 64])
    fc1 = tf.nn.relu(tf.add(tf.matmul(fc1, W_fc1), B_fc1))
    fc1 = tf.nn.dropout(fc1, keep_prob)

    # 输出层
    W_fc2 = weight_variable([1024, CAPTCHA_LEN * CHAR_SET_LEN], 'W_fc2')
    B_fc2 = bias_variable([CAPTCHA_LEN * CHAR_SET_LEN], 'B_fc2')
    output = tf.add(tf.matmul(fc1, W_fc2), B_fc2, 'output')

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=output))
    optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

    predict = tf.reshape(output, [-1, CAPTCHA_LEN, CHAR_SET_LEN], name='predict')
    labels = tf.reshape(Y, [-1, CAPTCHA_LEN, CHAR_SET_LEN], name='labels')

    # 预测结果
    # 请注意 predict_max_idx 的 name，在测试model时会用到它
    predict_max_idx = tf.argmax(predict, axis=2, name='predict_max_idx')
    labels_max_idx = tf.argmax(labels, axis=2, name='labels_max_idx')
    predict_correct_vec = tf.equal(predict_max_idx, labels_max_idx)
    accuracy = tf.reduce_mean(tf.cast(predict_correct_vec, tf.float32))

    # 开始深度学习
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        steps = 0
        for epoch in range(6000):
            train_data, train_label = get_next_batch(64, 'train', steps)
            sess.run(optimizer, feed_dict={X: train_data, Y: train_label, keep_prob: 0.75})
            if steps % 100 == 0:
                test_data, test_label = get_next_batch(100, 'validate', steps)
                acc = sess.run(accuracy, feed_dict={X: test_data, Y: test_label, keep_prob: 1.0})
                now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("steps=%d, accuracy=%f, now_time=%s" % (steps, acc, now_time))
                if acc > 0.999:  # 当准确度大于99.9%的时候停止并保存模型
                    saver.save(sess, MODEL_SAVE_PATH + model_file_name, global_step=steps)
                    break
            steps += 1


if __name__ == '__main__':
    image_filename_list, total = get_image_file_name(CAPTCHA_IMAGE_PATH)
    random.seed(time.time())
    # 打乱顺序
    random.shuffle(image_filename_list)
    trainImageNumber = int(total * TRAIN_IMAGE_PERCENT)

    # 分成测试集
    TRAINING_IMAGE_NAME = image_filename_list[: trainImageNumber]

    # 和验证集
    VALIDATION_IMAGE_NAME = image_filename_list[trainImageNumber:]
    train_data_with_cnn()
    print('Training finished')
