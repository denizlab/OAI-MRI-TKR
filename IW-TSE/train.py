#!/usr/bin/env python3
import h5py
import os.path
import numpy as np
import pandas as pd
import math
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import tensorflow as tf
#from sklearn.model_selection import StratifiedKFold
from ModelResnet3D import generate_model
from DataGenerator import DataGenerator

#from keras.models import Sequential
#from keras.optimizers import SGD, Adam
#from keras.layers import Dropout, Dense, Conv3D, MaxPooling3D, GlobalAveragePooling3D, Activation, BatchNormalization,Flatten
from keras.callbacks import LearningRateScheduler, TensorBoard, EarlyStopping, ModelCheckpoint, Callback
from sklearn.metrics import roc_auc_score

tf.app.flags.DEFINE_boolean('batch_norm', True, 'Use BN or not')
tf.app.flags.DEFINE_float('lr', 0.0001, 'Initial learning rate.')
tf.app.flags.DEFINE_integer('filters_in_last', 128, 'Number of filters on the last layer')
tf.app.flags.DEFINE_float('dr',0.0, 'Dropout rate when training.')
tf.app.flags.DEFINE_string('input_file','/gpfs/data/denizlab/Datasets/OAI/SAG_IW_TSE/dataBefore201807/data_SAG_IW_TSE.hdf5', 'File to read data')
tf.app.flags.DEFINE_string('file_path', '/gpfs/data/denizlab/Users/hrr288/Radiology_test/', 'Main Folder to Save outputs')
tf.app.flags.DEFINE_integer('val_fold', 1, 'Fold for cv')
tf.app.flags.DEFINE_string('file_folder','/gpfs/data/denizlab/Datasets/OAI/SAG_IW_TSE/', 'Path to HDF5 radiographs')

FLAGS = tf.app.flags.FLAGS

class auc_Histories(Callback):
    def on_train_begin(self, logs={}):
        self.aucs = []
        self.losses = []
 
    def on_train_end(self, logs={}):
        return
 
    def on_epoch_begin(self, epoch, logs={}):
        return
 
    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        y_pred = self.model.predict(self.model.validation_data)
        self.aucs.append(roc_auc_score(self.model.validation_data[1], y_pred))
        return
 
    def on_batch_begin(self, batch, logs={}):
        return
 
    def on_batch_end(self, batch, logs={}):
        return



class roc_callback(Callback):
    def __init__(self,index,val_fold):
        _params = {'dim': (384,384,36),
              'batch_size': 4,
              'n_classes': 2,
              'n_channels': 1,
              'shuffle': False,
              'normalize' : True,
              'randomCrop' : False,
              'randomFlip' : False,
              'flipProbability' : -1}
        self.x = DataGenerator(directory = '/gpfs/data/denizlab/Users/hrr288/TSE_dataset/Fold_'+str(val_fold)+'/CV_'+str(index)+'_train.csv',file_folder=FLAGS.file_folder,  **_params)
        self.x_val = DataGenerator(directory = '/gpfs/data/denizlab/Users/hrr288/TSE_dataset/Fold_'+str(val_fold)+'/CV_'+str(index)+'_val.csv',file_folder=FLAGS.file_folder,  **_params)
        self.y = pd.read_csv('/gpfs/data/denizlab/Users/hrr288/TSE_dataset/Fold_'+str(val_fold)+'/CV_'+str(index)+'_train.csv').Label
        self.y_val = pd.read_csv('/gpfs/data/denizlab/Users/hrr288/TSE_dataset/Fold_'+str(val_fold)+'/CV_'+str(index)+'_val.csv').Label
        self.auc = []
        self.val_auc = []
        self.losses = []
        self.val_losses = []
    
    def on_train_begin(self, logs={}):
        return
 
    def on_train_end(self, logs={}):
        return
 
    def on_epoch_begin(self, epoch, logs={}):
        return
    
    def on_epoch_end(self, epoch, logs={}):        
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        y_pred = self.model.predict_generator(self.x)
        y_true = self.y[:len(y_pred)]
        roc = roc_auc_score(y_true, y_pred)      
        
        y_pred_val = self.model.predict_generator(self.x_val)
        y_true_val = self.y_val[:len(y_pred_val)]
        roc_val = roc_auc_score(y_true_val, y_pred_val)      
        self.auc.append(roc)
        self.val_auc.append(roc_val)
        #print(len(y_true),len(y_true_val))
        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc,4)),str(round(roc_val,4))),end=100*' '+'\n')
        return
 
    def on_batch_begin(self, batch, logs={}):
        return
 
    def on_batch_end(self, batch, logs={}):
        return   
    
'''
    Def: Code to plot loss curves
    Params: history = keras output from training
            loss_path = path to save curve
'''
def plot_loss_curves(history, loss_path): #, i):
    f = plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    #plt.show()    
    #path = '/data/kl2596/curves/loss/' + loss_path + '.jpeg'
    f.savefig(loss_path)


'''
    Def: Code to plot accuracy curves
    Params: history = keras output from training
            acc_path = path to save curve
'''
def plot_accuracy_curves(history, acc_path): #, i):
    f = plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    #plt.show() 
    #path = '/data/kl2596/curves/accuracy/' + acc_path + '.jpeg'
    f.savefig(acc_path)


def plot_auc_curves(auc_history, acc_path): #, i):
    f = plt.figure()
    plt.plot(auc_history.auc)
    plt.plot(auc_history.val_auc)
    plt.title('model AUC')
    plt.ylabel('auc')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    #plt.show() 
    #path = '/data/kl2596/curves/accuracy/' + acc_path + '.jpeg'
    f.savefig(acc_path)

def train_model(model, train_data, val_data, path, index,val_fold):
    #model.summary()
    
    # Early Stopping callback that can be found on Keras website
    
    # Create path to save weights with model checkpoint
    weights_path = path + 'weights-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}-{loss:.2f}-{acc:.2f}.hdf5'
    model_checkpoint = ModelCheckpoint(weights_path, monitor = 'val_loss', save_best_only = True, 
                                       verbose=0, period=1)
    
    # Save loss and accuracy curves using Tensorboard
    tensorboard_callback = TensorBoard(log_dir = path, 
                                       histogram_freq = 0, 
                                       write_graph = False, 
                                       write_grads = False, 
                                       write_images = False)
    
    auc_history = roc_callback(index,val_fold)
    #es = EarlyStopping(monitor='val_auc', mode='max', verbose=1, patience=50)
    callbacks_list = [model_checkpoint, tensorboard_callback, auc_history]
    #es = EarlyStopping(monitor='val_auc', mode='max', verbose=1, patience=150)
    history = model.fit_generator(generator = train_data, validation_data = val_data, epochs=10, 
                        #use_multiprocessing=True, workers=6, 
                        callbacks = callbacks_list)
    
    accuracy = auc_history.val_auc
    print('*****************************')
    print('best auc:',np.max(accuracy))
    print('average auc:',np.mean(accuracy))
    print('*****************************') 

    accuracy = history.history['val_acc']
    print('*****************************')
    print('best acc:',np.max(accuracy))
    print('average acc:',np.mean(accuracy))
    print('*****************************')
    loss_path = path + 'loss_curve.jpeg'
    acc_path = path + 'acc_curve.jpeg'
    auc_path = path + 'auc_curve.jpeg'
    plot_loss_curves(history, loss_path)
    plot_accuracy_curves(history, acc_path)
    plot_auc_curves(auc_history, auc_path)
    #model.save_weights(weights_path)
   
    
'''
    Def: Code to run stratified cross validation to train my network
    Params: num_of_folds = number of folds to cross validate
            lr = learning rate
            dr = dropout rate
            filters_in_last = number of filters in last convolutional layer (we tested 64 and 128)
            batch_norm = True or False for batch norm in model
            data = MRI images
            labels = labels corresponding to MRI images
            file_path = path to save network weights, curves, and tensorboard callbacks
'''
def cross_validation(val_fold, lr, dr, filters_in_last, file_path):
    train_params = {'dim': (384,384,36),
          'batch_size': 4,
          'n_classes': 2,
          'n_channels': 1,
          'shuffle': True,
          'normalize' : True,
          'randomCrop' : True,
          'randomFlip' : True,
          'flipProbability' : -1}
    
    val_params = {'dim': (384,384,36),
          'batch_size': 4,
          'n_classes': 2,
          'n_channels': 1,
          'shuffle': False,
          'normalize' : True,
          'randomCrop' : False,
          'randomFlip' : False,
          'flipProbability' : -1} 
    
    model_path = file_path + 'Tnetres_Best/lr24ch32kerne773773_strde222_new_arch/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
            

    num_of_folds = 6
    for i in range(num_of_folds):
        model = generate_model(learning_rate = 2 * 10 **(-4))
        model.summary()
        print(train_params)
        #print(train_index, test_index)
        print('Running Fold', i+1, '/', num_of_folds)   
        fold_path = model_path + 'Fold_' + str(val_fold) + '/CV_'+str(i+1)+'/'
        print(fold_path)
        
        if not os.path.exists(fold_path):
            os.makedirs(fold_path)    
        
        training_generator = DataGenerator(directory = '/gpfs/data/denizlab/Users/hrr288/TSE_dataset/Fold_'+str(val_fold)+'/CV_'+str(i+1)+'_train.csv',file_folder=FLAGS.file_folder,  **train_params)
        validation_generator = DataGenerator(directory = '/gpfs/data/denizlab/Users/hrr288/TSE_dataset/Fold_'+str(val_fold)+'/CV_'+str(i+1)+'_val.csv',file_folder=FLAGS.file_folder,  **val_params)
        
        train_model(model=model, 
                    train_data = training_generator,
                    val_data = validation_generator,
                    path = fold_path, index = i+1,val_fold=val_fold)



def main(argv=None):
    print('Begin training for fold ',FLAGS.val_fold)
    cross_validation(val_fold=FLAGS.val_fold, 
                     lr=FLAGS.lr, dr=FLAGS.dr, filters_in_last=FLAGS.filters_in_last,   
                     file_path = FLAGS.file_path)

if __name__ == "__main__":
    tf.app.run()


