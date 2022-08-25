#!/usr/bin/env python3

from keras.models import Sequential
from keras.optimizers import SGD, Adam
#from keras.layers import Dropout, Dense, Conv3D, MaxPooling3D,GlobalMaxPooling3D, GlobalAveragePooling3D, Activation, BatchNormalization,Flatten
from resnet3d import Resnet3DBuilder
import tensorflow as tf
def generate_model(learning_rate = 2 * 10 **(-4)):
    #auc = tf.keras.metrics.AUC()
    #model = Resnet3DBuilder.build_resnet_50((384, 384, 160, 1), 1)
    model = Resnet3DBuilder.build_resnet_18((352, 352, 35, 1), 1)
    model.compile(loss='binary_crossentropy',
                      metrics = ['accuracy'],
                      optimizer = Adam(lr=learning_rate,beta_1=0.99, beta_2=0.999))#SGD(lr=1e-2, momentum = 0.9))
    return model

