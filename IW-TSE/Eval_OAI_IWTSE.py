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
import os

from sklearn import metrics


from keras.models import load_model

from Augmentation import RandomCrop, CenterCrop, RandomFlip



from sklearn.metrics import roc_auc_score,auc,roc_curve,average_precision_score

import tensorflow as tf


from DataGenerator import DataGenerator as DG

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
        self.file_folder = file_folder+"00m/"
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
            pre_image = h5py.File(self.file_folder+ ID, "r")['data/'].value.astype('float64')
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

tf.app.flags.DEFINE_string('model_path', '/gpfs/data/denizlab/Users/hrr288/Radiology_test/Tnetres_Best/lr24ch32kerne773773_strde222_new_arch/', 'Folder with the models')
tf.app.flags.DEFINE_string('val_csv_path', '/gpfs/data/denizlab/Users/hrr288/TSE_dataset/', 'Folder with the fold splits')

tf.app.flags.DEFINE_string('test_csv_path', '/gpfs/data/denizlab/Users/hrr288/data/OAI_SAG_TSE_test.csv', 'Folder with the test csv')
tf.app.flags.DEFINE_string('result_path', './', 'Folder to save output csv with preds')
tf.app.flags.DEFINE_bool('vote', False, 'Choice to generate binary predictions for each model to compute final sensitivity/specificity')
tf.app.flags.DEFINE_string('file_folder','/gpfs/data/denizlab/Datasets/OAI/SAG_IW_TSE/', 'Path to HDF5 radiographs of test set')
tf.app.flags.DEFINE_string('train_file_folder','/gpfs/data/denizlab/Datasets/OAI/SAG_IW_TSE/', 'Path to IW TSE HDF5 radiographs of OAI train/val set')


FLAGS = tf.app.flags.FLAGS
def main(argv=None):




    val_params = {'dim': (384,384,36),
              'batch_size': 1,
              'n_classes': 2,
              'n_channels': 1,
              'shuffle': False,
              'normalize' : True,
              'randomCrop' : False,
              'randomFlip' : False,
              'flipProbability' : -1}



    validation_generator = DataGenerator(directory = FLAGS.test_csv_path,file_folder=FLAGS.file_folder,  **val_params)
    df = pd.read_csv(FLAGS.test_csv_path,index_col=0)

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
    AUCS = []
    preds = []
    dfs = []
    pred_arr = np.zeros(df.shape[0])
    for i in np.arange(1,8):
        
        for j in np.arange(1,7):
            model = load_model(base_path+'/'+models['fold_'+str(i)][j-1])
            if FLAGS.vote:
                test_df = pd.read_csv(FLAGS.val_csv_path+'Fold_'+str(i)+'/CV_'+str(j)+'_val.csv')
                test_generator = DG(directory = FLAGS.val_csv_path+'Fold_'+str(i)+'/CV_'+str(j)+'_val.csv',file_folder=FLAGS.train_file_folder,  **val_params)

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




    # In[ ]:


    df["Preds"] = pred_arr
    if  FLAGS.vote:
        df.to_csv(FLAGS.result_path+"OAI_results_vote.csv")
    else:
        df.to_csv(FLAGS.result_path+"OAI_results.csv")
if __name__ == "__main__":
    tf.app.run()

