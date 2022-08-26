# OAI_MRI_train

## DESS, IW-TSE and T1-TSE training/evaluation

To train fold in {1,2,..,7} run

python3 train.py --file_path /path to save models/ --csv_path /path to csvs/ --file_folder /path to HDF5 radiograph files/ --val_fold fold
