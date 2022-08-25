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
import tensorflow as tf


from keras.models import load_model




from sklearn.metrics import roc_auc_score,auc,roc_curve,average_precision_score




from Augmentation import RandomCrop, CenterCrop, RandomFlip

from DataGenerator import DataGenerator
tf.app.flags.DEFINE_string('model_path', '/gpfs/data/denizlab/Users/hrr288/Radiology_test/SAG3D_lr24_18_stride221_kernel777773/', 'Folder with the models')
tf.app.flags.DEFINE_string('csv_path', '/gpfs/data/denizlab/Users/hrr288/Tianyu_dat/TestSets/', 'Folder with the fold splits')
tf.app.flags.DEFINE_string('result_path', './', 'Folder to save output csv with preds')
tf.app.flags.DEFINE_string('file_folder','/gpfs/data/denizlab/Datasets/OAI/SAG_3D_DESS/', 'Path to HDF5 radiographs')


FLAGS = tf.app.flags.FLAGS

def main(argv=None):


    base_path = FLAGS.model_path
    csv_path = FLAGS.csv_path

    # Choosing the model in each folder with lowest val loss

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

    val_params = {'dim': (352,352,144),
              'batch_size': 1,
              'n_classes': 2,
              'n_channels': 1,
              'shuffle': False,
              'normalize' : True,
              'randomCrop' : False,
              'randomFlip' : False,
              'flipProbability' : -1,
              'cropDim' : (352,352,144)}
    


    dfs = []
    
    for i in np.arange(1,8):
        print("Fold_"+str(i))
        validation_generator = DataGenerator(directory = csv_path+'Fold_'+str(i)+'/Fold_'+str(i)+'_test.csv',file_folder=FLAGS.file_folder, **val_params)
        df = pd.read_csv(csv_path+'/Fold_'+str(i)+'/Fold_'+str(i)+'_test.csv')
        pred_arr = np.zeros(df.shape[0])
        
        for j in np.arange(1,7):
            model = load_model(base_path+'/'+models['fold_'+str(i)][j-1])
            
            s = model.predict_generator(validation_generator)
            
            pred_arr += np.squeeze(s)
        pred_arr = pred_arr/6
        df["Preds"] = pred_arr
        dfs.append(df)

        
        
        
            
        

    full_df = pd.concat(dfs)
    full_df.to_csv(FLAGS.result_path+"OAI_DESS_results.csv")
if __name__ == "__main__":
    tf.app.run()


