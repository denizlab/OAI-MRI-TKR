# OAI_MRI_train

## DESS training/evaluation

To train fold in {1,2,..,7} in nested Cross validation, run

python3 train.py --file_path /path to save models/ --csv_path /path to split csvs/DESS/ --file_folder /path to DESS HDF5 radiograph files/ --val_fold fold

To evaluate on the matched OAI cohort, run

python3 evaluate.py --model_path /path to the saved models/ --csv_path /path to split csvs/DESS/ --file_folder /path to DESS HDF5 radiograph files/ 

To evaluate on the internal OAI testset, run

python3 Eval_OAI_DESS.py --model_path /path to the saved models/ --val_csv_path /path to split csvs/DESS/ --test_csv_path /path to csvs/OAI_SAG_DESS_test.csv/ --file_folder /path to DESS HDF5 radiograph files/ --vote True/False 

## IW-TSE training/evaluation

To train fold in {1,2,..,7} in nested Cross validation, run

python3 train.py --file_path /path to save models/ --csv_path /path to split csvs/IW-TSE/ --file_folder /path to IW-TSE HDF5 radiograph files/ --val_fold fold

To evaluate on the matched OAI cohort, run

python3 evaluate.py --model_path /path to the saved models/ --csv_path /path to split csvs/IW-TSE/ --file_folder /path to IW-TSE HDF5 radiograph files/ 

To evaluate on the internal OAI testset, run 

python3 Eval_OAI_IWTSE.py --model_path /path to the saved models/ --val_csv_path /path to split csvs/IW-TSE/ --test_csv_path /path to csvs/OAI_SAG_TSE_test.csv/ --file_folder /path to IW-TSE HDF5 radiograph files/ --vote True/False 

## T1-TSE training/evaluation

To train fold in {1,2,..,7} in nested Cross validation, run

python3 train.py --file_path /path to save models/ --csv_path /path to split csvs/T1-TSE/ --file_folder /path to T1-TSE HDF5 radiograph files/ --val_fold fold

To evaluate on the matched OAI cohort, run

python3 evaluate.py --model_path /path to the saved models/ --csv_path /path to split csvs/T1-TSE/ --file_folder /path to T1-TSE HDF5 radiograph files/ 

To evaluate on the internal OAI testset, run 

python3 Eval_OAI_T1TSE.py --model_path /path to the saved models/ --val_csv_path /path to split csvs/T1-TSE/ --test_csv_path /path to csvs/OAI_COR_TSE_test.csv/ --file_folder /path to T1-TSE HDF5 radiograph files/ --vote True/False 
