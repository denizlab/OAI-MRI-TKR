#!/usr/bin/env python3

from keras.models import Sequential,Model
from keras.optimizers import SGD, Adam
from keras.layers import Input#, Dropout, Dense, Conv3D, MaxPooling3D, GlobalMaxPooling3D ,GlobalAveragePooling3D, Activation, BatchNormalization,Flatten
#from keras.models import Sequential
#from keras.layers import Dropout, Dense, Conv3D, MaxPooling3D,GlobalMaxPooling3D, GlobalAveragePooling3D, Activation, BatchNormalization,Flatten
from resnet3d import Resnet3DBuilder
#from keras.regularizers import l2


def generate_model(learning_rate = 1 * 10 **(-4)):
    
    print('**************************')
    
    
    model = Resnet3DBuilder.build_resnet_18((48, 96, 18, 32), 1)
    model.compile(loss='binary_crossentropy',
                      metrics = ['accuracy'],
                      optimizer = Adam(lr=learning_rate,beta_1=0.99, beta_2=0.999))#SGD(lr=1e-2, momentum = 0.9))
    return model

    #input1 = Input(shape=(384, 384, 36, 1))
    #input2 = Input(shape=(352, 352, 144, 1))

    #out = Resnet3DBuilder.build_resnet_18((48, 96, 18, 32),input1,input2,1)
       
    #merged_model = Model([input1, input2], out)
    
    merged_model.compile(loss='binary_crossentropy',
                      metrics = ['accuracy'],
                      optimizer = Adam(lr=learning_rate,beta_1=0.99, beta_2=0.999))#SGD(lr=1e-2, momentum = 0.9))
    #loss='categorical_crossentropy'

    return merged_model





