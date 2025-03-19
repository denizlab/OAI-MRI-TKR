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
    def __init__(self, directory1,directory2,file_folder1,file_folder2, batch_size=6, dim1=(384,384,36), dim2= (352,352,144), n_channels=1, n_classes=10, shuffle=True,normalize = True, randomCrop = True, randomFlip = True,flipProbability = -1):
        'Initialization'
        self.dim1 = dim1
        self.dim2 = dim2
        self.dim3 = (384,384,144)
        self.batch_size = batch_size
        self.dataset = pd.read_csv(directory1)
        self.IWdataset = pd.read_csv(directory1)
        self.DESSdataset = pd.read_csv(directory2)
        #self.list_IDs = list_IDs
        self.list_IDs = pd.read_csv(directory1)['ID']
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
        X, y = self.__data_generation(indexes)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X1 = np.empty((self.batch_size, *self.dim1, self.n_channels))
        X2 = np.empty((self.batch_size, *self.dim3, self.n_channels))
        #X2 = np.empty((self.batch_size, 6))
        y = np.empty((self.batch_size), dtype=int)
        for i in range(len(indexes)):
            # Store sample
            #print(i,ID)
            filename1 = self.IWdataset.iloc[indexes[i]]['h5Name']
            filename2 = self.DESSdataset.iloc[indexes[i]]['h5Name'] 
            pre_image1 = h5py.File(self.file_folder1 + filename1, "r")['data/'].value.astype('float64')
            pre_image2 = h5py.File(self.file_folder2 + filename2, "r")['data/'].value.astype('float64')
            #pre_image = padding_image(data = image,shape = [448,448,48])
            #pre_image = np.zeros(image.shape)
            #pre_image = image
            if pre_image1.shape[2] < 36:
                pre_image1 = padding_image(data = pre_image1)
            if pre_image2.shape[2] < 144:
                pre_image2 = padding_image2(data = pre_image2)
            # normalize
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
            y[i] = self.IWdataset.iloc[indexes[i]].Label

        return [X1,X2], y

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
def normalize_MRIs(image):
    mean = np.mean(image)
    std = np.std(image)
    image -= mean
    #image -= 95.09
    image /= std
    #image /= 86.38
    return image

tf.app.flags.DEFINE_string('model_path','/gpfs/data/denizlab/Users/hrr288/Radiology_test/TCBmodelv1_400_add_final_arch/Dnetv1/add_ch32/', 'Folder with the models')
tf.app.flags.DEFINE_string('val_csv_path', '/gpfs/data/denizlab/Users/hrr288/Tianyu_dat/TestSets/', 'Folder with the fold splits')

tf.app.flags.DEFINE_string('test_csv_path1', '/gpfs/data/denizlab/Users/hrr288/data/OAI_SAG_TSE_test.csv', 'Folder with IW TSE test csv')
tf.app.flags.DEFINE_string('test_csv_path2', '/gpfs/data/denizlab/Users/hrr288/data/OAI_SAG_DESS_test.csv', 'Folder with DESS test csv')

tf.app.flags.DEFINE_string('result_path', './', 'Folder to save output csv with preds')
tf.app.flags.DEFINE_bool('vote', False, 'Choice to generate binary predictions for each model to compute final sensitivity/specificity')
tf.app.flags.DEFINE_string('file_folder1','/gpfs/data/denizlab/Datasets/OAI/SAG_IW_TSE/', 'Path to IW TSE HDF5 radiographs of test set')
tf.app.flags.DEFINE_string('file_folder2','/gpfs/data/denizlab/Datasets/OAI/SAG_3D_DESS/', 'Path to DESS  HDF5 radiographs of test set')
tf.app.flags.DEFINE_string('IWdataset_csv','/gpfs/data/denizlab/Datasets/OAI/SAG_IW_TSE/HDF5_00_cohort_2_prime.csv', 'Path to HDF5_00_cohort_2_prime.csv')

tf.app.flags.DEFINE_string('DESSdataset_csv','/gpfs/data/denizlab/Datasets/OAI/SAG_3D_DESS/HDF5_00_SAG_3D_DESScohort_2_prime.csv', 'Path to HDF5_00_SAG_3D_DESScohort_2_prime.csv')



FLAGS = tf.app.flags.FLAGS
def main(argv=None):




    val_params = {'dim1': (384,384,36),
                  'dim2': (352,352,144),
                  'batch_size': 1,
                  'n_classes': 1,
                  'n_channels': 1,
                  'shuffle': False,
                  'normalize' : True,
                  'randomCrop' : False,
                  'randomFlip' : False,
                  'flipProbability' : -1,
                     }



    validation_generator = DataGenerator(directory1 = FLAGS.test_csv_path1,directory2 = FLAGS.test_csv_path2,file_folder1=FLAGS.file_folder1,file_folder2=FLAGS.file_folder2,  **val_params)
    df = pd.read_csv(FLAGS.test_csv_path2,index_col=0)

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
                test_generator = DG(directory = FLAGS.val_csv_path+'Fold_'+str(i)+'/CV_'+str(j)+'_val.csv',file_folder1=FLAGS.file_folder1,file_folder2=FLAGS.file_folder2,IWdataset_csv=FLAGS.IWdataset_csv,DESSdataset_csv=FLAGS.DESSdataset_csv,  **val_params)

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

