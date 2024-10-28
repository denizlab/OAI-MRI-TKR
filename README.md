# Prediction of total knee replacement using deep learning analysis of knee MRI

## Project Overview
This project implements deep learning models to predict total knee replacement using MRI analysis. Our models achieve up to 0.90 AUC using a combination of MRI and radiograph data, significantly outperforming traditional methods. The implementation supports multiple MRI sequence types (DESS, IW-TSE, T1-TSE) and includes a novel multi-input model architecture.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Model Weights](#model-weights)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
  - [DESS Sequence](#dess-sequence)
  - [IW-TSE Sequence](#iw-tse-sequence)
  - [T1-TSE Sequence](#t1-tse-sequence)
  - [Multi-Input Model](#multi-input-model)
- [Results](#results)
- [Citation](#citation)

## Prerequisites
- Python 3.6
- CUDA-compatible GPU (recommended)
- HDF5 dataset files
- Required Python packages (install via `environment.yaml`)

## Installation
1. Clone this repository:
```bash
git clone https://github.com/denizlab/OAI-MRI-TKR.git
```

2. Set up the Python environment:
```bash
conda env create -f environment.yaml
conda activate tkr_mri_env
```

## Model Weights
Pre-trained weights from our 7-fold nested cross-validation are available at:
[https://zenodo.org/records/11237172](https://zenodo.org/records/11237172)

## Data Preparation
### Dataset Requirements
- OAI (Osteoarthritis Initiative) dataset access required
- MOST (Multicenter Osteoarthritis Study) dataset access required
- Contact respective organizations for data access permissions
### Data Organization
- Separate HDF5 files for each MRI sequence type (DESS, IW-TSE, T1-TSE)
- CSV files for dataset splits and cohort information
- Proper file structure for OAI and MOST datasets

## Usage

### DESS Sequence

#### Training
```bash
python3 ./DESS/train.py \
    --file_path /path/to/save/models \
    --csv_path /path/to/split/csvs/DESS \
    --file_folder /path/to/DESS/HDF5/files \
    --val_fold [1-7]
```

#### Evaluation Options

1. Matched OAI Cohort:
```bash
python3 ./DESS/evaluate.py \
    --model_path /path/to/saved/models \
    --csv_path /path/to/split/csvs/DESS \
    --file_folder /path/to/DESS/HDF5/files
```

2. Internal OAI Test Set:
```bash
python3 ./DESS/Eval_OAI_DESS.py \
    --model_path /path/to/saved/models \
    --val_csv_path /path/to/split/csvs/DESS \
    --test_csv_path /path/to/csvs/OAI_SAG_DESS_test.csv \
    --file_folder /path/to/DESS/HDF5/files \
    --vote [True/False]
```

3. MOST Test Set:
```bash
python3 ./DESS/Eval_MOST_DESS.py \
    --model_path /path/to/saved/models \
    --val_csv_path /path/to/split/csvs/DESS \
    --test_csv_path /path/to/csvs/MOST_MRI_test.csv \
    --train_file_folder /path/to/DESS/HDF5/files \
    --file_folder /path/to/MOST/HDF5/files \
    --vote [True/False] \
    --contrast [HR_COR_STIR/SAG_PD_FAT_SAT]
```

### IW-TSE Sequence

#### Training
```bash
python3 ./IW-TSE/train.py \
    --file_path /path/to/save/models \
    --csv_path /path/to/split/csvs/IW-TSE \
    --file_folder /path/to/IW-TSE/HDF5/files \
    --val_fold [1-7]
```

#### Evaluation Options

1. Matched OAI Cohort:
```bash
python3 ./IW-TSE/evaluate.py \
    --model_path /path/to/saved/models \
    --csv_path /path/to/split/csvs/IW-TSE \
    --file_folder /path/to/IW-TSE/HDF5/files
```

2. Internal OAI Test Set:
```bash
python3 ./IW-TSE/Eval_OAI_IWTSE.py \
    --model_path /path/to/saved/models \
    --val_csv_path /path/to/split/csvs/IW-TSE \
    --test_csv_path /path/to/csvs/OAI_SAG_TSE_test.csv \
    --file_folder /path/to/IW-TSE/HDF5/files \
    --vote [True/False]
```

3. MOST Test Set:
```bash
python3 ./IW-TSE/Eval_MOST_IWTSE.py \
    --model_path /path/to/saved/models \
    --val_csv_path /path/to/split/csvs/IW-TSE \
    --test_csv_path /path/to/csvs/MOST_MRI_test.csv \
    --train_file_folder /path/to/IW-TSE/HDF5/files \
    --file_folder /path/to/MOST/HDF5/files \
    --vote [True/False] \
    --contrast [HR_COR_STIR/SAG_PD_FAT_SAT]
```

### T1-TSE Sequence

#### Training
```bash
python3 ./T1-TSE/train.py \
    --file_path /path/to/save/models \
    --csv_path /path/to/split/csvs/T1-TSE \
    --file_folder /path/to/T1-TSE/HDF5/files \
    --val_fold [1-7]
```

#### Evaluation Options

1. Matched OAI Cohort:
```bash
python3 ./T1-TSE/evaluate.py \
    --model_path /path/to/saved/models \
    --csv_path /path/to/split/csvs/T1-TSE \
    --file_folder /path/to/T1-TSE/HDF5/files
```

2. Internal OAI Test Set:
```bash
python3 ./T1-TSE/Eval_OAI_T1TSE.py \
    --model_path /path/to/saved/models \
    --val_csv_path /path/to/split/csvs/T1-TSE \
    --test_csv_path /path/to/csvs/OAI_COR_TSE_test.csv \
    --file_folder /path/to/T1-TSE/HDF5/files \
    --vote [True/False]
```

3. MOST Test Set:
```bash
python3 ./T1-TSE/Eval_MOST_T1TSE.py \
    --model_path /path/to/saved/models \
    --val_csv_path /path/to/split/csvs/T1-TSE \
    --test_csv_path /path/to/csvs/MOST_MRI_test.csv \
    --train_file_folder /path/to/T1-TSE/HDF5/files \
    --file_folder /path/to/MOST/HDF5/files \
    --vote [True/False] \
    --contrast [HR_COR_STIR/SAG_PD_FAT_SAT]
```

### Multi-Input Model

#### Training
```bash
python3 ./MI-DESS_IWTSE/train.py \
    --file_path /path/to/save/models \
    --csv_path /path/to/split/csvs/DESS \
    --file_folder1 /path/to/IW-TSE/HDF5/files \
    --file_folder2 /path/to/DESS/HDF5/files \
    --val_fold [1-7] \
    --IWdataset_csv /path/to/csvs/HDF5_00_cohort_2_prime.csv \
    --DESSdataset_csv /path/to/csvs/HDF5_00_SAG_3D_DESScohort_2_prime.csv
```

#### Evaluation Options

1. Matched OAI Cohort:
```bash
python3 ./MI-DESS_IWTSE/evaluate.py \
    --model_path /path/to/saved/models \
    --csv_path /path/to/split/csvs/DESS \
    --file_folder1 /path/to/IW-TSE/HDF5/files \
    --file_folder2 /path/to/DESS/HDF5/files \
    --IWdataset_csv /path/to/csvs/HDF5_00_cohort_2_prime.csv \
    --DESSdataset_csv /path/to/csvs/HDF5_00_SAG_3D_DESScohort_2_prime.csv
```

2. Internal OAI Test Set:
```bash
python3 ./MI-DESS_IWTSE/Eval_OAI_MI.py \
    --model_path /path/to/saved/models \
    --val_csv_path /path/to/split/csvs/DESS \
    --test_csv_path1 /path/to/csvs/OAI_SAG_TSE_test.csv \
    --test_csv_path2 /path/to/csvs/OAI_SAG_DESS_test.csv \
    --file_folder1 /path/to/IW-TSE/HDF5/files \
    --file_folder2 /path/to/DESS/HDF5/files \
    --vote [True/False] \
    --IWdataset_csv /path/to/csvs/HDF5_00_cohort_2_prime.csv \
    --DESSdataset_csv /path/to/csvs/HDF5_00_SAG_3D_DESScohort_2_prime.csv
```

## Results

### Model Performance

**Table 1: Receiver operator characteristic analysis with areas under the curve (AUC) and area under the precision-recall curve (AUPRC) evaluating the diagnostic performance of the models to predict total knee replacement (TKR) using sevenfold nested cross-validation on the training a validation group in the OAI database.**

| Model | AUC (95% CI) | p value | AUPRC (95% CI) | Sensitivity (%) (95% CI) | Specificity (%) (95% CI) |
|-------|--------------|----------|----------------|-------------------------|-------------------------|
| **MLP model** |
| Traditional | 0.77 (0.74, 0.81) | Reference | 0.76 (0.71, 0.81) | 73 (68, 77) | 73 (68, 78) |
| **CNN models** |
| DESS | 0.88 (0.86, 0.91) | <0.001 | 0.87 (0.83, 0.91) | 82 (78, 86) | 81 (77, 85) |
| FS-IW-TSE | 0.86 (0.84, 0.89) | <0.001 | 0.87 (0.84, 0.90) | 77 (73,82) | 84 (80, 87) |
| Multi-input MRI | 0.85 (0.82, 0.88) | <0.001 | 0.85 (0.81, 0.89) | 79 (75, 83) | 79 (74, 83) |
| IW-TSE | 0.87 (0.84, 0.90) | <0.001 | 0.87 (0.84, 0.90) | 82 (78, 86) | 78 (73, 82) |
| Radiograph | 0.87 (0.84, 0.89) | <0.001 | 0.87 (0.84, 0.90) | 81 (76, 85) | 80 (76, 84) |
| **Ensemble models** |
| MRI | 0.89 (0.87, 0.91) | <0.001 | 0.89 (0.87, 0.91) | 79 (75, 83) | 86 (82, 89) |
| MRI and radiograph | 0.90 (0.87, 0.92) | <0.001 | 0.90 (0.87, 0.93) | 80 (76, 84) | 85 (81, 88) |

Key findings from our results:
- All CNN and ensemble models significantly outperformed the traditional MLP model (p < 0.001)
- The MRI and radiograph ensemble model achieved the highest performance with AUC of 0.90 (95% CI: 0.87-0.92)
- DESS sequence alone showed strong performance with AUC of 0.88 (95% CI: 0.86-0.91)



## Citation
If you use this code in your research, please cite our paper:
[https://www.nature.com/articles/s41598-023-33934-1](https://www.nature.com/articles/s41598-023-33934-1)

```bibtex
@article{rajamohan2023prediction,
   title={Prediction of total knee replacement using deep learning analysis of knee MRI},
   author={Rajamohan, Haresh Rengaraj and Wang, Tianyu and Leung, Kevin and Chang, Gregory and Cho, Kyunghyun and Kijowski, Richard and Deniz, Cem M},
   journal={Scientific reports},
   volume={13},
   number={1},
   pages={6922},
   year={2023},
   publisher={Nature Publishing Group UK London}
}
```