"""
ResUNet architecture in Keras TensorFlow
"""
import os
import numpy as np
import cv2

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from keras.layers import SeparableConv2D
# from keras.layers import ReLu

def squeeze_excite_block(inputs, ratio=8):
    init = inputs
    channel_axis = -1
    print("type of inputs pass to stem_block: ",type(init))
    print(init)
    xy = init.shape[channel_axis]
    print(type(xy))
    print(xy)
    filters = xy#.values
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    x = Multiply()([init, se])
    return x

def stem_block(x, n_filter, strides):
    x_init = x

    ## Conv 1
    x = Conv2D(n_filter, (3, 3), padding="same", strides=strides)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(n_filter, (3, 3), padding="same")(x)

    ## Shortcut
    s  = Conv2D(n_filter, (1, 1), padding="same", strides=strides)(x_init)
    s = BatchNormalization()(s)

    ## Add
    x = Add()([x, s])
    print("type of x passed to squeeze_excite_block: ",x)
    print(x)
    x = squeeze_excite_block(x)
    return x


def resnet_block(x, n_filter, strides=1):
    x_init = x

    ## Conv 1
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(n_filter, (3, 3), padding="same", strides=strides)(x)
    ## Conv 2
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(n_filter, (3, 3), padding="same", strides=1)(x)

    ## Shortcut
    s  = Conv2D(n_filter, (1, 1), padding="same", strides=strides)(x_init)
    s = BatchNormalization()(s)

    ## Add
    x = Add()([x, s])
    x = squeeze_excite_block(x)
    return x

def aspp_block(x, num_filters, rate_scale=1):
    x1 = Conv2D(num_filters, (3, 3), dilation_rate=(6 * rate_scale, 6 * rate_scale), padding="SAME")(x)
    x1 = BatchNormalization()(x1)

    x2 = Conv2D(num_filters, (3, 3), dilation_rate=(12 * rate_scale, 12 * rate_scale), padding="SAME")(x)
    x2 = BatchNormalization()(x2)

    x3 = Conv2D(num_filters, (3, 3), dilation_rate=(18 * rate_scale, 18 * rate_scale), padding="SAME")(x)
    x3 = BatchNormalization()(x3)

    x4 = Conv2D(num_filters, (3, 3), padding="SAME")(x)
    x4 = BatchNormalization()(x4)

    x5 = Conv2D(num_filters, (3, 3), dilation_rate=(24 * rate_scale, 24 * rate_scale), padding="SAME")(x)
    x5 = BatchNormalization()(x5)

    y = Add()([x1, x2, x3,x5, x4])
    y = Conv2D(num_filters, (1, 1), padding="SAME")(y)
    return y


def deep_aspp_block(x, num_filters, rate_scale=1):
  # x1 = ZeroPadding2D(padding=(6 * rate_scale,6 * rate_scale))(x)
  x1 = SeparableConv2D(num_filters , (3,3), strides = (1,1) , padding = 'same', dilation_rate = (1,1) , activation = None)(x)
  x1 = BatchNormalization()(x1)
  x1 = ReLU()(x1)
  # x1 = Reshape((32,32,256))(x1)

    # x1 = Conv2D(num_filters, (3, 3), dilation_rate=(6 * rate_scale, 6 * rate_scale), padding="SAME")(x)
    # x1 = BatchNormalization()(x1)

  # x2 = Conv2D(num_filters, (3, 3), dilation_rate=(12 * rate_scale, 12 * rate_scale), padding="SAME")(x)
  # x2 = BatchNormalization()(x2)

  # x3 = Conv2D(num_filters, (3, 3), dilation_rate=(18 * rate_scale, 18 * rate_scale), padding="SAME")(x)
  # x3 = BatchNormalization()(x3)
  # x2 = ZeroPadding2D(padding=(12 * rate_scale,12 * rate_scale))(x)
  x2 = SeparableConv2D(num_filters , (3,3), strides = (1,1) , padding = 'same', dilation_rate = (1,1) , activation = None)(x)
  x2 = BatchNormalization()(x2)
  x2 = ReLU()(x2)
  # x2 = Reshape((32,32,256))(x2)

  # x3 = ZeroPadding2D(padding=(18 * rate_scale,18 * rate_scale))(x)
  x3 = SeparableConv2D(num_filters , (3,3), strides = (1,1) , padding = 'same', dilation_rate = (1,1) , activation = None)(x)
  x3 = BatchNormalization()(x3)
  x3 = ReLU()(x3)
  # x3 = Reshape((32,32,256))(x3)

  x4 = Conv2D(num_filters, (3, 3), padding="SAME")(x)
  x4 = BatchNormalization()(x4)

  y = Add()([x1, x2, x3, x4])
  y = Conv2D(num_filters, (1, 1), padding="SAME")(y)
  return y

def daspp_block(x, num_filters, rate_scale=1):
  print("x shape: ", x.shape)

  x0 = SeparableConv2D(num_filters, (1, 1), padding="SAME")(x)
  x0 = BatchNormalization()(x0)
  print("x0 shape: ", x0.shape)

  x1 = SeparableConv2D(num_filters, (3, 3), dilation_rate=(3 * rate_scale, 3 * rate_scale), padding="SAME")(x)
  # x1 = ReLu()(x1)
  x1 = tf.keras.layers.Activation(tf.nn.relu)(x1)
  x1 = BatchNormalization()(x1)
  x1 = Conv2D(num_filters, (3, 3), padding="SAME")(x1)
  print("x1 shape: ", x1.shape)

  x2 = SeparableConv2D(num_filters, (3, 3), dilation_rate=(6 * rate_scale, 6 * rate_scale), padding="SAME")(x)
  # x2 = ReLu()(x2)
  x2 = tf.keras.layers.Activation(tf.nn.relu)(x2)
  x2 = BatchNormalization()(x2)
  x2 = Conv2D(num_filters, (3, 3), padding="SAME")(x2)
  print("x2 shape: ", x2.shape)

  x3 = SeparableConv2D(num_filters, (3, 3), dilation_rate=(9 * rate_scale, 9 * rate_scale), padding="SAME")(x)
  # x3 = ReLu()(x3)
  x3 = tf.keras.layers.Activation(tf.nn.relu)(x3)
  x3 = BatchNormalization()(x3)
  x3 = Conv2D(num_filters, (3, 3), padding="SAME")(x3)
  print("x3 shape: ", x3.shape)
  

  x4 = AveragePooling2D(pool_size=(2, 2), padding= 'same')(x)
  x4 = Conv2D(16, (1, 1), padding="SAME")(x4)
  x4 = UpSampling2D(size=(2, 2))(x4)
  print("x4 shape: ", x4.shape)

  # y = Add()([x0 , x1, x2, x3, x4])
  # y = Add()([x0 , x1, x2, x3])
  y = Concatenate()([x , x0 , x1, x2, x3, x4])
  y = Conv2D(num_filters, (1, 1), padding="SAME")(y)
  return y

def attetion_block(g, x):
    """
        g: Output of Parallel Encoder block
        x: Output of Previous Decoder block
    """
    xy = x.shape[-1]
    print(xy)
    filters = xy#.value

    g_conv = BatchNormalization()(g)
    g_conv = Activation("relu")(g_conv)
    g_conv = Conv2D(filters, (3, 3), padding="SAME")(g_conv)

    g_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(g_conv)

    x_conv = BatchNormalization()(x)
    x_conv = Activation("relu")(x_conv)
    x_conv = Conv2D(filters, (3, 3), padding="SAME")(x_conv)

    gc_sum = Add()([g_pool, x_conv])

    gc_conv = BatchNormalization()(gc_sum)
    gc_conv = Activation("relu")(gc_conv)
    gc_conv = Conv2D(filters, (3, 3), padding="SAME")(gc_conv)

    gc_mul = Multiply()([gc_conv, x])
    return gc_mul

class ResUnetPlusPlus:
    def __init__(self, input_size=256):
        self.input_size = input_size

    def build_model(self):
        n_filters = [16, 32, 64, 128, 256]
        inputs = Input((self.input_size, self.input_size, 3))

        c0 = inputs
        c1 = stem_block(c0, n_filters[0], strides=1)

        ## Encoder
        c2 = resnet_block(c1, n_filters[1], strides=2)
        c3 = resnet_block(c2, n_filters[2], strides=2)
        c4 = resnet_block(c3, n_filters[3], strides=2)

        ## Bridge
        b1 = daspp_block(c4, n_filters[4])

        ## Decoder
        d1 = attetion_block(c3, b1)
        d1 = UpSampling2D((2, 2))(d1)
        d1 = Concatenate()([d1, c3])
        d1 = resnet_block(d1, n_filters[3])

        d2 = attetion_block(c2, d1)
        d2 = UpSampling2D((2, 2))(d2)
        d2 = Concatenate()([d2, c2])
        d2 = resnet_block(d2, n_filters[2])

        d3 = attetion_block(c1, d2)
        d3 = UpSampling2D((2, 2))(d3)
        d3 = Concatenate()([d3, c1])
        d3 = resnet_block(d3, n_filters[1])

        ## output
        outputs = daspp_block(d3, n_filters[0])
        outputs = Conv2D(1, (1, 1), padding="same")(outputs)
        outputs = Activation("sigmoid")(outputs)

        ## Model
        model = Model(inputs, outputs)
        return model