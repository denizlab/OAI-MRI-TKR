# ==============================================================================
# Copyright (C) 2023 Haresh Rengaraj Rajamohan, Tianyu Wang, Kevin Leung, 
# Gregory Chang, Kyunghyun Cho, Richard Kijowski & Cem M. Deniz 
#
# This file is part of OAI-MRI-TKR
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# ==============================================================================
import six
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten,
    GlobalMaxPooling3D,
    GlobalAveragePooling3D
)
from keras.layers import Dropout, Dense, Conv3D, MaxPooling3D,GlobalMaxPooling3D, GlobalAveragePooling3D, Activation, BatchNormalization,Flatten

from keras.layers.convolutional import (
    Conv3D,
    AveragePooling3D,
    MaxPooling3D
)
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
from math import ceil

kernel_regularizer_value = 5e-5

def _bn_relu(input):
    """Helper to build a BN -> relu block (copy-paster from
    raghakot/keras-resnet)
    """
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)


def _conv_bn_relu3D(**conv_params):
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1, 1))
    kernel_initializer = conv_params.setdefault(
        "kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer",
                                                l2(kernel_regularizer_value))

    def f(input):
        conv = Conv3D(filters=filters, kernel_size=kernel_size,
                      strides=strides, #kernel_initializer=kernel_initializer,
                      padding=padding,
                      kernel_regularizer=kernel_regularizer)(input)
        return _bn_relu(conv)

    return f


def _bn_relu_conv3d(**conv_params):
    """Helper to build a  BN -> relu -> conv3d block.
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer",
                                                "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer",
                                                l2(kernel_regularizer_value))

    def f(input):
        activation = _bn_relu(input)

        return Conv3D(filters=filters, kernel_size=kernel_size,
                      strides=strides, #kernel_initializer=kernel_initializer,
                      padding=padding,
                      kernel_regularizer=kernel_regularizer)(activation)
    return f


def _shortcut3d(input, residual):
    """3D shortcut to match input and residual and merges them with "sum" at
    the same time
    """
    stride_dim1 = ceil(input._keras_shape[DIM1_AXIS]*1.0 \
        / residual._keras_shape[DIM1_AXIS])
    stride_dim2 = ceil(input._keras_shape[DIM2_AXIS]*1.0 \
        / residual._keras_shape[DIM2_AXIS])
    stride_dim3 = ceil(input._keras_shape[DIM3_AXIS]*1.0 \
        / residual._keras_shape[DIM3_AXIS])
    equal_channels = residual._keras_shape[CHANNEL_AXIS] \
        == input._keras_shape[CHANNEL_AXIS]
    

    shortcut = input
    if stride_dim1 > 1 or stride_dim2 > 1 or stride_dim3 > 1 \
            or not equal_channels:
        shortcut = Conv3D(
            filters=residual._keras_shape[CHANNEL_AXIS],
            kernel_size=(1, 1, 1),
            strides=(stride_dim1, stride_dim2, stride_dim3),
            #kernel_initializer="he_normal", 
            padding="valid",
            kernel_regularizer=l2(kernel_regularizer_value)
            )(input)
    return add([shortcut, residual])


def _residual_block3d(block_function, filters, repetitions,
                      is_first_layer=False):
    def f(input):
        for i in range(repetitions):
            strides = (1, 1, 1)
            if i == 0 and not is_first_layer:
                strides = (2, 2, 2)
            input = block_function(filters=filters,
                                   strides=strides,
                                   is_first_block_of_first_layer=(
                                       is_first_layer and i == 0)
                                   )(input)
        return input

    return f


def basic_block(filters, strides=(1, 1, 1),
                is_first_block_of_first_layer=False):
    """Basic 3 X 3 X 3 convolution blocks. Extended from raghakot's 2D impl.
    """
    def f(input):
        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Conv3D(filters=filters, kernel_size=(3, 3, 3),
                           strides=strides, #kernel_initializer="he_normal",
                           padding="same", kernel_regularizer=l2(kernel_regularizer_value)
                           )(input)
        else:
            conv1 = _bn_relu_conv3d(filters=filters,
                                    kernel_size=(3, 3, 3),
                                    strides=strides
                                    )(input)

        residual = _bn_relu_conv3d(filters=filters, kernel_size=(3, 3, 3),
                                   )(conv1)

        return _shortcut3d(input, residual)

    return f


def bottleneck(filters, strides=(1, 1, 1),
               is_first_block_of_first_layer=False):
    """Basic 3 X 3 X 3 convolution blocks. Extended from raghakot's 2D impl.
    """
    def f(input):
        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv_1_1 = Conv3D(filters=filters, kernel_size=(1, 1, 1),
                              strides=strides, padding="same",
                              #kernel_initializer="he_normal",
                              kernel_regularizer=l2(kernel_regularizer_value))(input)
        else:
            conv_1_1 = _bn_relu_conv3d(filters=filters, kernel_size=(1, 1, 1),
                                       strides=strides)(input)

        conv_3_3 = _bn_relu_conv3d(filters=filters,
                                   kernel_size=(3, 3, 3))(conv_1_1)
        residual = _bn_relu_conv3d(filters=filters * 4,
                                   kernel_size=(1, 1, 1))(conv_3_3)
        return _shortcut3d(input, residual)

    return f


def _handle_data_format():
    global DIM1_AXIS
    global DIM2_AXIS
    global DIM3_AXIS
    global CHANNEL_AXIS
    if K.image_data_format() == 'channels_last':
        DIM1_AXIS = 1
        DIM2_AXIS = 2
        DIM3_AXIS = 3
        CHANNEL_AXIS = 4
    else:
        CHANNEL_AXIS = 1
        DIM1_AXIS = 2
        DIM2_AXIS = 3
        DIM3_AXIS = 4


def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier


class Resnet3DBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs,  block_fn, repetitions):
        """Instantiate a vanilla ResNet3D keras model
        # Arguments
            input_shape: Tuple of input shape in the format
            (conv_dim1, conv_dim2, conv_dim3, channels) if dim_ordering='tf'
            (filter, conv_dim1, conv_dim2, conv_dim3) if dim_ordering='th'
            num_outputs: The number of outputs at the final softmax layer
            block_fn: Unit block to use {'basic_block', 'bottlenack_block'}
            repetitions: Repetitions of unit blocks
        # Returns
            model: a 3D ResNet model that takes a 5D tensor (volumetric images
            in batch) as input and returns a 1D vector (prediction) as output.
        """

        _handle_data_format()
        if len(input_shape) != 4:
            raise Exception("Input shape should be a tuple "
                            "(conv_dim1, conv_dim2, conv_dim3, channels) "
                            "for tensorflow as backend or "
                            "(channels, conv_dim1, conv_dim2, conv_dim3) "
                            "for theano as backend")
        block_fn = _get_block(block_fn)
        input = Input(shape=input_shape)
        filters = 32
        # first conv
        conv1 = _conv_bn_relu3D(filters=filters, kernel_size=(7, 7, 3),
                                strides=(2, 2, 1))(input)
        pool1 = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 1),
                             padding="same")(conv1)
        
        conv2 = _conv_bn_relu3D(filters=filters, kernel_size=(7, 7, 3),
                                strides=(2, 2, 1))(pool1)
        #conv3 = _conv_bn_relu3D(filters=32, kernel_size=(3, 3, 3),
        #                        strides=(2, 2, 2))(conv2)
        #pool2 = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2),
        #                     padding="same")(conv2)
        # repeat blocks
        block = conv2
        #filters = 32
        for i, r in enumerate(repetitions):
            block = _residual_block3d(block_fn, filters=filters,
                                      repetitions=r, is_first_layer=(i == 0)
                                      )(block)
            filters *= 2

        # last activation
        block_output = _bn_relu(block)

        # average poll and classification
        #pool3 = AveragePooling3D(pool_size=(block._keras_shape[DIM1_AXIS],
        #                                    block._keras_shape[DIM2_AXIS],
        #                                    block._keras_shape[DIM3_AXIS]),
        #                         strides=(1, 1, 1))(block_output)
        #flatten1 = Flatten()(pool3)
        flatten1 = GlobalAveragePooling3D()(block_output)
        if num_outputs > 1:
            dense = Dense(units=num_outputs,
                          #kernel_initializer="he_normal",
                          activation="softmax",
                          kernel_regularizer=l2(kernel_regularizer_value))(flatten1)
        else:
            dense = Dense(units=num_outputs,
                          #kernel_initializer="he_normal",
                          activation="sigmoid",
                          kernel_regularizer=l2(kernel_regularizer_value))(flatten1)

        model = Model(inputs=input, outputs=dense)
        return model

    @staticmethod
    def build_resnet_18(input_shape, num_outputs):
        return Resnet3DBuilder.build(input_shape, num_outputs, basic_block,
                                     [2, 2, 2, 2])

    @staticmethod
    def build_resnet_34(input_shape, num_outputs):
        return Resnet3DBuilder.build(input_shape, num_outputs, basic_block,
                                     [3, 4, 6, 3])

    @staticmethod
    def build_resnet_50(input_shape, num_outputs):
        return Resnet3DBuilder.build(input_shape, num_outputs, bottleneck,
                                     [3, 4, 6, 3])


    @staticmethod
    def build_resnet_101(input_shape, num_outputs):
        return Resnet3DBuilder.build(input_shape, num_outputs, bottleneck,
                                     [3, 4, 23, 3])

    @staticmethod
    def build_resnet_152(input_shape, num_outputs):
        return Resnet3DBuilder.build(input_shape, num_outputs, bottleneck,
                                     [3, 8, 36, 3])
