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
import pandas as pd
import h5py
import nibabel as nib
import keras
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colorbar
import matplotlib.colors
import pandas as pd
import numpy as np

from sklearn import metrics

import os
import argparse


from keras.models import load_model
from skimage.transform import resize

import tensorflow as tf


from Augmentation import RandomCrop, CenterCrop, RandomFlip



from sklearn.metrics import roc_auc_score,auc,roc_curve,average_precision_score

from DataGenerator import DataGenerator as dg


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, directory,file_folder, batch_size=6, dim=(384,384,36),contrast="HR_COR_STIR" ,n_channels=1, n_classes=10, shuffle=True,normalize = True, randomCrop = True, randomFlip = True,flipProbability = -1):
        'Initialization'
        self.dim = dim
        
        self.batch_size = batch_size
        self.dataset = pd.read_csv(directory)
        #self.list_IDs = list_IDs
        self.contrast = contrast
        self.list_IDs = pd.read_csv(directory)['MOST_ID']
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.file_folder = file_folder+contrast+"/V0/"
        self.normalize = normalize
        self.randomCrop = randomCrop
        self.randomFlip = randomFlip
        self.flipProbability = flipProbability
        self.side = {0:"LEFT",1:"RIGHT"}


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        # Generate data
        X, y = self.__data_generation(indexes)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    def normalize_MRIs(self,image):
        mean = np.mean(image)
        std = np.std(image)
        image -= mean
        #image -= 95.09
        image /= std
        #image /= 86.38
        return image
    def padding_image(self,data,dim):
        
        if dim==2:
            l,w,h = data.shape
            images = np.zeros((l,w,36))
            zstart = int(np.ceil((36-data.shape[2])/2))

            images[:,:,zstart:zstart + h] = data
        elif dim==1:
            l,w,h = data.shape
            images = np.zeros((l,self.dim[1],h))
            ystart = int(np.ceil((self.dim[1]-data.shape[1])/2))

            images[:,ystart:ystart + w,:] = data
        elif dim==0:
            l,w,h = data.shape
            images = np.zeros((self.dim[0],w,h))
            xstart = int(np.ceil((self.dim[0]-data.shape[0])/2))

            images[xstart:xstart + l,:,:] = data


        return images
    
    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        #X2 = np.empty((self.batch_size, 6))
        y = np.empty((self.batch_size), dtype=int)
        for i in range(len(indexes)):
            # Store sample
            #print(i,ID)
            filename =  self.dataset.iloc[indexes[i]]["MOST_ID"]+"_"+self.dataset.iloc[indexes[i]]["ACROSTIC"]+"_V0_"+self.side[self.dataset.iloc[indexes[i]]["KNEE"]]+"_"+self.contrast+".hdf5"
            pre_image = h5py.File(self.file_folder + filename, "r")["data"][:].astype('float64')
            #pre_image = padding_image(data = image,shape = [448,448,48])
            #pre_image = np.zeros(image.shape)
            #pre_image = image
            if pre_image.shape[2]<36:
                pre_image = self.padding_image(pre_image,dim=2)
            
            pre_image = resize(pre_image, (416, 416,self.dim[2]),anti_aliasing=True)
            
            if self.normalize:
                pre_image = self.normalize_MRIs(pre_image)
            # Augmentation
            if self.randomFlip:
                pre_image = RandomFlip(image=pre_image,p=0.5).horizontal_flip(p=self.flipProbability)
            if self.randomCrop:
                pre_image = RandomCrop(pre_image).crop_along_hieght_width_depth(self.dim)
            else:
                pre_image = CenterCrop(image=pre_image).crop(size = self.dim)

            tempx = np.zeros([1,self.dim[0],self.dim[1],self.dim[2],1])
            tempx[0,:,:,:,0] = pre_image
            X[i] = tempx
            
            #X1[i,:,:,:,0] = pre_image1
            #X2[i,:,:,:,0] = pre_image2
            #X2[i] = self.dataset[self.dataset.FileName == ID].iloc[:,-6:]
            # Store class
            #print(self.dataset[self.dataset.FileName == ID].Label)
            y[i] = self.dataset.iloc[indexes[i]].TKRlabel

        return X, y

    def getXvalue(self,index):
        return self.__getitem__(index)

def padding_image(data):
    l,w,h = data.shape
    images = np.zeros((l,w,36))
    zstart = int(np.ceil((36-data.shape[2])/2))
    images[:,:,zstart:zstart + h] = data
    return images 

def padding_image2(data):
    l,w,h = data.shape
    images = np.zeros((l,w,144))
    zstart = int(np.ceil((144-data.shape[2])/2))
    images[:,:,zstart:zstart + h] = data
    return images 




tf.app.flags.DEFINE_string('model_path', '/gpfs/data/denizlab/Users/hrr288/Radiology_test/Tnetres_Best/lr24ch32kerne773773_strde222_new_arch/', 'Folder with the models')
tf.app.flags.DEFINE_string('val_csv_path', '/gpfs/data/denizlab/Users/hrr288/TSE_dataset/', 'Folder with the fold splits')

tf.app.flags.DEFINE_string('test_csv_path', '/gpfs/data/denizlab/Users/hrr288/data/MOST_radiology/MOST_MRI_test.csv', 'Folder with the test csv')
tf.app.flags.DEFINE_string('result_path', './', 'Folder to save output csv with preds')
tf.app.flags.DEFINE_bool('vote', False, 'Choice to generate binary predictions for each model to compute final sensitivity/specificity')
tf.app.flags.DEFINE_string('file_folder','/gpfs/data/denizlab/Datasets/MOST/', 'Path to HDF5 radiographs of MOST set')
tf.app.flags.DEFINE_string('train_file_folder','/gpfs/data/denizlab/Datasets/OAI/SAG_IW_TSE/', 'Path to HDF5 radiographs of OAI train/val set')

tf.app.flags.DEFINE_string('contrast', 'HR_COR_STIR', 'MOST contrast to evaluate')


FLAGS = tf.app.flags.FLAGS


def main() -> None:
    # argparser
    base_path = FLAGS.model_path

    models= {'fold_1':[],'fold_2':[],'fold_3':[],'fold_4':[],'fold_5':[],'fold_6':[],'fold_7':[]}
    for fold in np.arange(1,8):
        tmp_mod_list = []
        for cv in np.arange(1,7):
            dir_1 = 'Fold_'+str(fold)+'/CV_'+str(cv)+'/'
            files_avai =  os.listdir(base_path+dir_1)
            cands = []
            cands_score = []
            for fs in files_avai:
                if 'weights' not in fs:
                    continue
                else:
                    
                    cands_score.append(float(fs.split('-')[2]))
                    cands.append(dir_1+fs)
            ind_c = int(np.argmin(cands_score))
            
            tmp_mod_list.append(cands[ind_c])
        models['fold_'+str(fold)]=tmp_mod_list



    val_params = {'dim': (384,384,36),
              'batch_size': 1,
              'contrast': FLAGS.contrast,
              'n_classes': 2,
              'n_channels': 1,
              'shuffle': False,
              'normalize' : True,
              'randomCrop' : False,
              'randomFlip' : False,
              'flipProbability' : -1}

    test_params = {'dim': (384,384,36),
              'batch_size': 1,
              'n_classes': 2,
              'n_channels': 1,
              'shuffle': False,
              'normalize' : True,
              'randomCrop' : False,
              'randomFlip' : False,
              'flipProbability' : -1}

    

    validation_generator = DataGenerator(directory = FLAGS.test_csv_path, file_folder=FLAGS.file_folder, **val_params)
    df = pd.read_csv(FLAGS.test_csv_path,index_col=0)

    AUCS = []
    preds = []
    dfs = []
    pred_arr = np.zeros(df.shape[0])
    for i in np.arange(1,8):
        
        print("Fold_"+str(i))
        
        for j in np.arange(1,7):
            model = load_model(base_path+'/'+models['fold_'+str(i)][j-1])
            if FLAGS.vote:
                test_df = pd.read_csv(FLAGS.val_csv_path+'/Fold_'+str(i)+'/CV_'+str(j)+'_val.csv')
                test_generator = dg(directory = FLAGS.val_csv_path+'/Fold_'+str(i)+'/CV_'+str(j)+'_val.csv',file_folder=FLAGS.train_file_folder,  **test_params)

                test_pred = model.predict_generator(test_generator)
                test_df["Pred"] = test_pred
                fpr, tpr, thresholds = metrics.roc_curve(test_df["Label"], test_df["Pred"])
                opt_ind = np.argmax(tpr-fpr)
                opt_thresh = thresholds[int(opt_ind)]
                s = model.predict_generator(validation_generator)

                pred_arr += (np.squeeze(s)>=opt_thresh)
            else:
                s = model.predict_generator(validation_generator)
                pred_arr += np.squeeze(s)


        
        #AUCS.append(roc_auc_score(df['Label'],pred_arr))
        
        #preds.extend(list(pred_arr))
        
            
    pred_arr = pred_arr/42

    df["Preds"] = pred_arr
    if FLAGS.vote:
      df.to_csv("MOST_DESS_"+FLAGS.contrast+"_results_vote.csv")
    else:
      df.to_csv("MOST_DESS_"+FLAGS.contrast+"_results.csv")


if __name__ == "__main__":
    main()




