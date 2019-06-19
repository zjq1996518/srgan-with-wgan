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

        model.compile(optimizer=keras.optimizers.RMSprop(2e-5), loss=self.wasserstein_loss)

        return model

    def build_vgg_model(self):

        img = Input(shape=[None, None, 3])
        vgg = VGG19(include_top=False, weights='./vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')
        vgg.outputs = [vgg.layers[20].output]
        vgg54 = Model(inputs=img, outputs=vgg(img))
        vgg54.trainable = False

        return vgg54

    def build_srgan(self):

        self.discriminator.trainable = False

        valid_img = Input(shape=[None, None, 3])

        fake_input = Input(shape=[None, None, 3])
        fake_img = self.generator(fake_input)

        vgg_out1 = self.vgg_model(valid_img)
        vgg_out2 = self.vgg_model(fake_img)

        vgg_mse_loss = K.mean((vgg_out1 - vgg_out2) ** 2)

        model = Model([valid_img, fake_input], self.discriminator(fake_img))

        model.compile(keras.optimizers.RMSprop(2e-5),
                      loss=lambda y_true, y_pred: vgg_mse_loss + 10e-3 * K.mean(y_true * y_pred))

        return model

    def get_train_data():

        files = None
        root = None
        for _root, dirs, _files in os.walk('/home/notebooks/imagenet'):
            files = _files
            root = _root

        x_train = []
        x_resize = []

        for file in files:

            img = cv2.imread(os.path.join(root, file))[:, :, [2, 1, 0]]
            shape = img.shape
            if shape[2] != 3:
                continue

            if shape[1] % 2 != 0:
                img = cv2.resize(img, (shape[1] - 1, shape[0]))
            if shape[0] % 2 != 0:
                img = cv2.resize(img, (shape[1], shape[0] - 1))

            img_resize = cv2.resize(img, (shape[1] // 2, shape[0] // 2))

            x_train.append(img)
            x_resize.append(img_resize)

            if len(x_train) == 1:
                x_train = np.array(x_train) / (255 / 2) - 1
                x_resize = np.array(x_resize) / (255 / 2) - 1

                yield x_train, x_resize

                x_train = []
                x_resize = []


exit = 0

if os.path.exists('./train.log'):
    os.remove('./train.log')

for t in range(4):

    if exit == 1:
        break

    data_generator = get_train_data()

    for i, (x_train, x_resize) in enumerate(data_generator):

        try:
            shape = x_train.shape

            valid = -np.ones([1, math.ceil(shape[1] / 16), math.ceil(shape[2] / 16), 1])
            fake = np.ones([1, math.ceil(shape[1] / 16), math.ceil(shape[2] / 16), 1])

            combine_loss = srGAN.model.train_on_batch([x_train, x_resize], valid)

            generate_imgs = srGAN.generator.predict(x_resize)
            d_loss_valid = srGAN.discriminator.train_on_batch(x_train, valid)
            d_loss_fake = srGAN.discriminator.train_on_batch(generate_imgs, fake)

            # 权重裁剪
            for l in srGAN.discriminator.layers:
                weights = l.get_weights()
                weights = [np.clip(w, -0.01, 0.01) for w in weights]
                l.set_weights(weights)

            with open('/home/notebooks/train.log', 'a', encoding='utf-8') as file:
                file.write('大循环' + str(t) + '第' + str(i) + '次 训练, 生成器损失：' + str(combine_loss) + ' 判别器损失: ' + str(
                    (d_loss_valid + d_loss_fake) / 2) + '\n')

            if i % 2000 == 0:
                generate_imgs = ((generate_imgs[0] + 1) * (255 / 2))[:, :, [2, 1, 0]]
                cv2.imwrite('./image/' + str(t) + '_' + str(i) + '.jpg', generate_imgs)

            if i % 1000 == 0:
                srGAN.discriminator.save_weights('./weight/discriminator_' + str(i) + '.w')
                srGAN.generator.save_weights('./weight/generator_' + str(i) + '.w')

        except KeyboardInterrupt as e:
            print('手动停止')
            exit = 1
            break

        except:
            with open('/home/notebooks/train.log', 'a', encoding='utf-8') as file:
                file.write('第' + str(i) + '条数据太大跳过\n')
