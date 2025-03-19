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





