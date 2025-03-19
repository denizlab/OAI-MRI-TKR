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
import numpy as np
import keras
import h5py
import pandas as pd
from Augmentation import RandomCrop, CenterCrop, RandomFlip

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, directory,file_folder, batch_size=8, dim=(384,384,36), n_channels=1,
                 n_classes=10, shuffle=True,normalize = True, randomCrop = True, randomFlip = True, 
                 flipProbability = -1):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.dataset = pd.read_csv(directory)
        #self.list_IDs = list_IDs
        self.list_IDs = pd.read_csv(directory)['h5Name']
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.file_folder = file_folder+'00m/'
        self.normalize = normalize
        self.randomCrop = randomCrop
        self.randomFlip = randomFlip
        self.flipProbability = flipProbability
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            #print(i,ID)
            pre_image = h5py.File(self.file_folder + ID, "r")['data/'].value.astype('float64')
            if pre_image.shape[2] < 36:
                pre_image = padding_image(data = pre_image)
            #pre_image = padding_image(data = image,shape = [448,448,48])
            #pre_image = np.zeros(image.shape)
            #pre_image = image
            # normalize
            if self.normalize:
                pre_image = normalize_MRIs(pre_image)
            # Augmentation
            if self.randomFlip:
                pre_image = RandomFlip(image=pre_image,p=0.5).horizontal_flip(p=self.flipProbability)
            if self.randomCrop:
                pre_image = RandomCrop(pre_image).crop_along_hieght_width_depth(self.dim)
            else:
                pre_image = CenterCrop(image=pre_image).crop(size = self.dim)
            
            X[i,:,:,:,0] = pre_image
            # Store class
            #print(self.dataset[self.dataset.FileName == ID].Label)
            y[i] = self.dataset[self.dataset.h5Name == ID].Label.unique()
        #print(y) 
        return X, y
    
    def getXvalue(self,index):
        return self.__getitem__(index)
    
def padding_image(data):
    l,w,h = data.shape
    images = np.zeros((l,w,36))
    zstart = int(np.ceil((36-data.shape[2])/2))
    
    images[:,:,zstart:zstart + h] = data
    return images

def normalize_MRIs(image):
    mean = np.mean(image)
    std = np.std(image)
    image -= mean
    #image -= 95.09
    image /= std
    #image /= 86.38
    return image
