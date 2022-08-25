import numpy as np
import keras
import h5py
import pandas as pd
from Augmentation import RandomCrop, CenterCrop, RandomFlip
from keras.models import load_model, Model
import tensorflow as tf

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, directory, file_folder1,file_folder2,IWdataset_csv,DESSdataset_csv, batch_size=6, dim1=(384,384,36), dim2= (352,352,144), n_channels=1, n_classes=10, shuffle=True,normalize = True, randomCrop = True, randomFlip = True,flipProbability = -1):
        'Initialization'
        self.dim1 = dim1
        self.dim2 = dim2
        self.dim3 = (384,384,144)
        self.batch_size = batch_size
        self.dataset = pd.read_csv(directory)
        self.IWdataset = pd.read_csv(IWdataset_csv)
        self.DESSdataset = pd.read_csv(DESSdataset_csv)
        #self.list_IDs = list_IDs
        self.list_IDs = pd.read_csv(directory)['ID']
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.file_folder1 = file_folder1+"00m/"
        self.file_folder2 = file_folder2+"00m/"
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
        X1 = np.empty((self.batch_size, *self.dim1, self.n_channels))
        X2 = np.empty((self.batch_size, *self.dim3, self.n_channels))
        #X2 = np.empty((self.batch_size, 6))
        y = np.empty((self.batch_size), dtype=int)
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            #print(i,ID)
            filename1 = self.IWdataset[self.IWdataset.ParticipantID == ID].FileName.values[0]
            filename2 = self.DESSdataset[self.DESSdataset.ParticipantID == ID].FileName.values[0]
            pre_image1 = h5py.File(self.file_folder1 + filename1, "r")['data/'].value.astype('float64')
            pre_image2 = h5py.File(self.file_folder2 + filename2, "r")['data/'].value.astype('float64')
            #pre_image = padding_image(data = image,shape = [448,448,48])
            #pre_image = np.zeros(image.shape)
            #pre_image = image
            if pre_image1.shape[2] < 36:
                pre_image1 = padding_image(data = pre_image1)
            # normalize
            if pre_image2.shape[2] < 144:
                pre_image2 = padding_image2(data = pre_image2)
            if self.normalize:
                pre_image1 = normalize_MRIs(pre_image1)
                pre_image2 = normalize_MRIs(pre_image2)
            # Augmentation
            if self.randomFlip:
                pre_image1 = RandomFlip(image=pre_image1,p=0.5).horizontal_flip(p=self.flipProbability)
                pre_image2 = RandomFlip(image=pre_image2,p=0.5).horizontal_flip(p=self.flipProbability)
            if self.randomCrop:
                pre_image1 = RandomCrop(pre_image1).crop_along_hieght_width_depth(self.dim1)
                pre_image2 = RandomCrop(pre_image2).crop_along_hieght_width_depth(self.dim2)
            else:
                pre_image1 = CenterCrop(image=pre_image1).crop(size = self.dim1)
                pre_image2 = CenterCrop(image=pre_image2).crop(size = self.dim2)

            tempx = np.zeros([1,384,384,36,1])
            tempx[0,:,:,:,0] = pre_image1
            X1[i] = tempx
            tempx = np.zeros([1,384,384,144,1])
            tempx[0,16:368,16:368,:,0] = pre_image2
            X2[i] = tempx
            #X1[i,:,:,:,0] = pre_image1
            #X2[i,:,:,:,0] = pre_image2
            #X2[i] = self.dataset[self.dataset.FileName == ID].iloc[:,-6:]
            # Store class
            #print(self.dataset[self.dataset.FileName == ID].Label)
            y[i] = self.dataset.iloc[i]["Label"]

        return [X1,X2], y

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

def padding_image2(data):
    l,w,h = data.shape
    images = np.zeros((l,w,144))
    zstart = int(np.ceil((144-data.shape[2])/2))
    images[:,:,zstart:zstart + h] = data
    return images 
