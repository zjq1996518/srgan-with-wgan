# encoding=utf8
import numpy as np
from tensorflow.keras.layers import *
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras import *
from tensorflow.keras.applications.vgg19 import VGG19
import tensorflow.keras.backend as K
import os
import cv2
import math


def conv_block(inputs, filters, strides):

    """
    判别器的卷积块,顺序为conv+bn+leaky_relu
    参数：
    filters: 卷积核个数
    strides: 卷积步长
    """

    block = Conv2D(kernel_size=3, filters=filters, strides=strides, padding='SAME')(inputs)
    block = BatchNormalization()(block)
    block = LeakyReLU()(block)

    return block


def residual_block(inputs):

    """
    生成器的残差块
    """

    block = Conv2D(kernel_size=3, filters=64, strides=1, padding='SAME')(inputs)
    block = BatchNormalization()(block)
    block = PReLU(shared_axes=[1, 2])(block)
    block = Conv2D(kernel_size=3, filters=64, strides=1, padding='SAME')(block)
    block = BatchNormalization()(block)
    block = Add()([inputs, block])

    return block


class SRGAN(object):
    """
    SRGAN 模型
    """

    def __init__(self):

        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.vgg_model = self.build_vgg_model()
        self.model = self.build_srgan()

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self, residual_count=8):

        """
        :param residual_count: 残差块个数
        :return: 生成器模型，默认8个残差层，生成的高分辨率图像宽高为原始的两倍
        """
        inputs = Input(shape=[None, None, 3])

        start_conv = Conv2D(kernel_size=9, filters=64, strides=1, padding='SAME')(inputs)
        start_conv = PReLU(shared_axes=[1, 2])(start_conv)

        b = residual_block(start_conv)
        for _ in range(residual_count - 1):
            b = residual_block(b)

        conv = Conv2D(kernel_size=9, filters=64, strides=1, padding='SAME')(b)
        conv = BatchNormalization()(conv)
        conv = Add()([start_conv, conv])

        out = Conv2D(kernel_size=3, filters=256, strides=1, padding='SAME')(conv)
        # 这里可以用上采样层替换
        out = Lambda(lambda x: tf.depth_to_space(x, 2))(out)
        out = PReLU(shared_axes=[1, 2])(out)

        out = Conv2D(kernel_size=9, filters=3, strides=1, padding='SAME', activation='tanh')(out)

        return Model(inputs, out)

    def build_discriminator(self):

        inputs = Input(shape=[None, None, 3])

        conv = Conv2D(kernel_size=3, filters=64, strides=1, padding='SAME')(inputs)
        conv = LeakyReLU()(conv)

        conv = conv_block(conv, 64, 2)

        for i in range(1, 4):
            conv = conv_block(conv, 128 * i, 1)
            conv = conv_block(conv, 128 * i, 2)

        dense = Dense(1024)(conv)
        dense = LeakyReLU()(dense)

        out = Dense(1)(dense)
        model = Model(inputs, out)

        # 判别器采用 wasserstein_loss
        model.compile(optimizer=keras.optimizers.RMSprop(2e-5), loss=self.wasserstein_loss)

        return model

    def build_vgg_model(self):

        img = Input(shape=[None, None, 3])
        vgg = VGG19(include_top=False)
        vgg.outputs = [vgg.layers[20].output]
        vgg54 = Model(inputs=img, outputs=vgg(img))
        vgg54.trainable = False

        return vgg54

    def build_srgan(self):

        self.discriminator.trainable = False

        # 真实图片
        valid_img = Input(shape=[None, None, 3])

        # 尺寸裁剪图片
        fake_input = Input(shape=[None, None, 3])

        # 生成器生成图片
        fake_img = self.generator(fake_input)

        # vgg编码
        vgg_out1 = self.vgg_model(valid_img)
        vgg_out2 = self.vgg_model(fake_img)

        # 计算vgg损失函数
        vgg_mse_loss = K.mean((vgg_out1 - vgg_out2) ** 2)

        model = Model([valid_img, fake_input], self.discriminator(fake_img))

        # 损失函数替换为 vgg_loss + wasserstein_loss
        model.compile(keras.optimizers.RMSprop(2e-5),
                      loss=lambda y_true, y_pred: vgg_mse_loss + 10e-3 * self.wasserstein_loss(y_true, y_pred))

        return model

    def get_train_data(self, batch=1):

        files = None
        root = None
        for _root, dirs, _files in os.walk('../data'):
            files = _files
            root = _root

        x_train = []
        x_resize = []

        for file in files:

            # 将opencv 读取图片转换为 rgb
            img = cv2.imread(os.path.join(root, file))[:, :, [2, 1, 0]]

            shape = img.shape
            if shape[2] != 3:
                continue

            # 如果图片宽高不为偶数，转换为偶数，否则生成器生成图片和原始图片宽高不匹配
            if shape[1] % 2 != 0:
                img = cv2.resize(img, (shape[1] - 1, shape[0]))
            if shape[0] % 2 != 0:
                img = cv2.resize(img, (shape[1], shape[0] - 1))

            img_resize = cv2.resize(img, (shape[1] // 2, shape[0] // 2))

            x_train.append(img)
            x_resize.append(img_resize)

            if len(x_train) == batch:
                x_train = np.array(x_train) / (255 / 2) - 1
                x_resize = np.array(x_resize) / (255 / 2) - 1

                yield x_train, x_resize

                x_train = []
                x_resize = []

    def train(self):

        should_exit = 0
        epochs = 2
        log_path = '../data/train.log'

        # 将训练信息记录到日志
        if os.path.exists(log_path):
            os.remove(log_path)

        for t in range(epochs):
            if should_exit == 1:
                break
            data_generator = self.get_train_data()

            for i, (x_train, x_resize) in enumerate(data_generator):

                try:
                    shape = x_train.shape

                    valid = -np.ones([1, math.ceil(shape[1] / 16), math.ceil(shape[2] / 16), 1])
                    fake = np.ones([1, math.ceil(shape[1] / 16), math.ceil(shape[2] / 16), 1])

                    combine_loss = self.model.train_on_batch([x_train, x_resize], valid)

                    generate_imgs = self.generator.predict(x_resize)
                    d_loss_valid = self.discriminator.train_on_batch(x_train, valid)
                    d_loss_fake = self.discriminator.train_on_batch(generate_imgs, fake)

                    # wgan权重裁剪
                    for l in self.discriminator.layers:
                        weights = l.get_weights()
                        weights = [np.clip(w, -0.01, 0.01) for w in weights]
                        l.set_weights(weights)

                    with open(log_path, 'a', encoding='utf-8') as file:
                        file.write(f'epoch {t} 第{i}个batch,生成器损失： {combine_loss} 判别器损失: {(d_loss_valid + d_loss_fake) / 2}\n')

                    # 每 2000 保存一次生成样本
                    if i % 2000 == 0:
                        generate_imgs = ((generate_imgs[0] + 1) * (255 / 2))[:, :, [2, 1, 0]]
                        cv2.imwrite(f'./image/{t}_{i}.jpg', generate_imgs)

                    # 每训练1000个batch保存权重
                    if i % 1000 == 0:
                        self.discriminator.save_weights('../weight/discriminator_{i}.w')
                        self.generator.save_weights('../weight/generator_{i}.w')

                except:
                    with open(log_path, 'a', encoding='utf-8') as file:
                        file.write('第{i}batch数据太大跳过\n')
