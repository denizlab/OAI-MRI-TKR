# OAI_MRI_train

## DESS training/evaluation

To train fold in {1,2,..,7} in nested Cross validation, run

python3 train.py --file_path /path to save models/ --csv_path /path to split csvs/DESS/ --file_folder /path to DESS HDF5 radiograph files/ --val_fold fold

To evaluate on the matched OAI cohort, run

python3 evaluate.py --model_path /path to the saved models/ --csv_path /path to split csvs/DESS/ --file_folder /path to DESS HDF5 radiograph files/ 

To evaluate on the internal OAI testset, run

python3 Eval_OAI_DESS.py --model_path /path to the saved models/ --val_csv_path /path to split csvs/DESS/ --test_csv_path /path to csvs/OAI_SAG_DESS_test.csv/ --file_folder /path to DESS HDF5 radiograph files/ --vote True/False 

To evaluate on the MOST testset, run 

python3 Eval_MOST_DESS.py --model_path /path to the saved models/ --val_csv_path /path to split csvs/DESS/ --test_csv_path /path to csvs/MOST_MRI_test.csv/ --train_file_folder /path to DESS HDF5 radiograph files/ --file_folder /Path to HDF5 radiographs of MOST set/ --vote True/False --contrast HR_COR_STIR/SAG_PD_FAT_SAT

## IW-TSE training/evaluation

To train fold in {1,2,..,7} in nested Cross validation, run

python3 train.py --file_path /path to save models/ --csv_path /path to split csvs/IW-TSE/ --file_folder /path to IW-TSE HDF5 radiograph files/ --val_fold fold

To evaluate on the matched OAI cohort, run

python3 evaluate.py --model_path /path to the saved models/ --csv_path /path to split csvs/IW-TSE/ --file_folder /path to IW-TSE HDF5 radiograph files/ 

To evaluate on the internal OAI testset, run 

python3 Eval_OAI_IWTSE.py --model_path /path to the saved models/ --val_csv_path /path to split csvs/IW-TSE/ --test_csv_path /path to csvs/OAI_SAG_TSE_test.csv/ --file_folder /path to IW-TSE HDF5 radiograph files/ --vote True/False 

To evaluate on the MOST testset, run 

python3 Eval_MOST_IWTSE.py --model_path /path to the saved models/ --val_csv_path /path to split csvs/IW-TSE/ --test_csv_path /path to csvs/MOST_MRI_test.csv/ --train_file_folder /path to IW-TSE HDF5 radiograph files/ --file_folder /Path to HDF5 radiographs of MOST set/ --vote True/False --contrast HR_COR_STIR/SAG_PD_FAT_SAT

## T1-TSE training/evaluation

To train fold in {1,2,..,7} in nested Cross validation, run

python3 train.py --file_path /path to save models/ --csv_path /path to split csvs/T1-TSE/ --file_folder /path to T1-TSE HDF5 radiograph files/ --val_fold fold

To evaluate on the matched OAI cohort, run

python3 evaluate.py --model_path /path to the saved models/ --csv_path /path to split csvs/T1-TSE/ --file_folder /path to T1-TSE HDF5 radiograph files/ 

To evaluate on the internal OAI testset, run 

python3 Eval_OAI_T1TSE.py --model_path /path to the saved models/ --val_csv_path /path to split csvs/T1-TSE/ --test_csv_path /path to csvs/OAI_COR_TSE_test.csv/ --file_folder /path to T1-TSE HDF5 radiograph files/ --vote True/False 

To evaluate on the MOST testset, run 

python3 Eval_MOST_T1TSE.py --model_path /path to the saved models/ --val_csv_path /path to split csvs/T1-TSE/ --test_csv_path /path to csvs/MOST_MRI_test.csv/ --train_file_folder /path to T1-TSE HDF5 radiograph files/ --file_folder /Path to HDF5 radiographs of MOST set/ --vote True/False --contrast HR_COR_STIR/SAG_PD_FAT_SAT


## Multi-Input Model training/evaluation

To train fold in {1,2,..,7} in nested Cross validation, run

python3 train.py --file_path /path to save models/ --csv_path /path to split csvs/DESS/ --file_folder1 /path to IW-TSE HDF5 radiograph files/ --file_folder2 /path to DESS HDF5 radiograph files/ --val_fold fold --IWdataset_csv /path to csvs/HDF5_00_cohort_2_prime.csv --DESSdataset_csv /path to csvs/HDF5_00_SAG_3D_DESScohort_2_prime.csv

To evaluate on the matched OAI cohort, run

python3 evaluate.py --model_path /path to the saved models/ --csv_path /path to split csvs/DESS/ --file_folder1 /path to IW-TSE HDF5 radiograph files/ --file_folder2 /path to DESS HDF5 radiograph files/ --IWdataset_csv /path to csvs/HDF5_00_cohort_2_prime.csv --DESSdataset_csv /path to csvs/HDF5_00_SAG_3D_DESScohort_2_prime.csv

To evaluate on the internal OAI testset, run

python3 Eval_OAI_MI.py --model_path /path to the saved models/ --val_csv_path /path to split csvs/DESS/ --test_csv_path1 /path to csvs/OAI_SAG_TSE_test.csv/ --test_csv_path2 /path to csvs/OAI_SAG_DESS_test.csv/ --file_folder1 /path to IW-TSE HDF5 radiograph files/ --file_folder2 /path to DESS HDF5 radiograph files/ --vote True/False --IWdataset_csv /path to csvs/HDF5_00_cohort_2_prime.csv --DESSdataset_csv /path to csvs/HDF5_00_SAG_3D_DESScohort_2_prime.csv

