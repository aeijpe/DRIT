#!/bin/bash
#SBATCH --job-name=run_DRIT_MMWHS                   # Job name
#SBATCH --partition=a6000                           # Partition
#SBATCH --qos=a6000_qos                             # Partition
#SBATCH --gpus-per-task=1                           # Number of gpus per node
#SBATCH --gpus=1                                    # Number of gpus in total
#SBATCH --ntasks=1                                  # Run on a single node
#SBATCH --cpus-per-task=16                          # Number of cores
#SBATCH --time=80:00:00                            # Time limit hrs:min:sec
#SBATCH --output=/projects/multimodal_disentanglement_review/DRIT/jobs/slurm_%j.log   
# Standard output and error log
pwd; hostname; date

# Source bashrc, such that the shell is setup properly
source ~/.bashrc
# Activate conda environment pyenv
source /home/a.eijpe/miniconda3/bin/activate
conda activate myenv
# export ITK_NIFTI_SFORM_PERMISSIVE=1

# Load cuda and cudnn (make sure versions match)
# eval `spack load --sh cuda@11.3 cudnn@8.2.0.53-11.3`

# Run your command
# Copy data to scratch before training

cd src
rsync -avv --info=progress2 --delete /projects/multimodal_disentanglement_review/nnUNet/nnUNet_preprocessed/Dataset033_MMWHSCT $SCRATCH/data_dir1
rsync -avv --info=progress2 --delete /projects/multimodal_disentanglement_review/nnUNet/nnUNet_preprocessed/Dataset034_MMWHSMRI $SCRATCH/data_dir2

# Train
python train.py --name mmwhs_run_fold_0 --batch_size 2 --data_dir1 $SCRATCH/data_dir1/Dataset033_MMWHSCT --data_dir2 $SCRATCH/data_dir2/Dataset034_MMWHSMRI --cases_folds 0 --data_type nnUNet --n_ep 10000 --splits_dir1 ../splits/splits_mmwhs_ct.json --splits_dir2 ../splits/splits_mmwhs_mri.json --model_save_freq 100 --img_save_freq 100 --display_freq 400
# python train.py --name mmwhs_run_fold_1 --batch_size 2 --data_dir1 $SCRATCH/data_dir1/Dataset033_MMWHSCT --data_dir2 $SCRATCH/data_dir2/Dataset034_MMWHSMRI --cases_folds 1 --data_type nnUNet --n_ep 1000 --splits_dir1 ../splits/splits_mmwhs_ct.json --splits_dir2 ../splits/splits_mmwhs_mri.json
# python train.py --name mmwhs_run_fold_2 --batch_size 2 --data_dir1 $SCRATCH/data_dir1/Dataset033_MMWHSCT --data_dir2 $SCRATCH/data_dir2/Dataset034_MMWHSMRI --cases_folds 2 --data_type nnUNet --n_ep 1000 --splits_dir1 ../splits/splits_mmwhs_ct.json --splits_dir2 ../splits/splits_mmwhs_mri.json
# python train.py --name mmwhs_run_fold_3 --batch_size 2 --data_dir1 $SCRATCH/data_dir1/Dataset033_MMWHSCT --data_dir2 $SCRATCH/data_dir2/Dataset034_MMWHSMRI --cases_folds 3 --data_type nnUNet --n_ep 1000 --splits_dir1 ../splits/splits_mmwhs_ct.json --splits_dir2 ../splits/splits_mmwhs_mri.json
# python train.py --name mmwhs_run_fold_4 --batch_size 2 --data_dir1 $SCRATCH/data_dir1/Dataset033_MMWHSCT --data_dir2 $SCRATCH/data_dir2/Dataset034_MMWHSMRI --cases_folds 4 --data_type nnUNet --n_ep 1000 --splits_dir1 ../splits/splits_mmwhs_ct.json --splits_dir2 ../splits/splits_mmwhs_mri.json

# # Evaluate
# python evaluate_model.py --name mmwhs_run_fold_0 --data_type nnUNet --cases_fold 0 --data_dir1 $SCRATCH/data_dir1/Dataset_033_MMWHSCT/ --data_dir2 $SCRATCH/data_dir2/Dataset_034_MMWHSMRI/
# python evaluate_model.py --name mmwhs_run_fold_1 --data_type nnUNet --cases_fold 1 --data_dir1 $SCRATCH/Dataset_033_MMWHSCT/ --data_dir2 $SCRATCH/Dataset_034_MMWHSMRI/
# python evaluate_model.py --name mmwhs_run_fold_2 --data_type nnUNet --cases_fold 2 --data_dir1 $SCRATCH/Dataset_033_MMWHSCT/ --data_dir2 $SCRATCH/Dataset_034_MMWHSMRI/
# python evaluate_model.py --name mmwhs_run_fold_3 --data_type nnUNet --cases_fold 3 --data_dir1 $SCRATCH/Dataset_033_MMWHSCT/ --data_dir2 $SCRATCH/Dataset_034_MMWHSMRI/
# python evaluate_model.py --name mmwhs_run_fold_4 --data_type nnUNet --cases_fold 4 --data_dir1 $SCRATCH/Dataset_033_MMWHSCT/ --data_dir2 $SCRATCH/Dataset_034_MMWHSMRI/

# # Create fake dataset 
# python create_fake_img.py --resume results/mmwhs_run_fold_0/.pth --name mmwhs_run_fold_0 --a2b 1 --data_dir1 ../../../data/MMWHS/CT_withGT_proc/ --cases_folds 0 --result_dir ../../../data/other/fake_MR/ --data_type nnUNET
# python create_fake_img.py --resume results/mmwhs_run_fold_1/.pth --name mmwhs_run_fold_1 --a2b 1 --data_dir1 ../../../data/MMWHS/CT_withGT_proc/ --cases_folds 1 --result_dir ../../../data/other/fake_MR/ --data_type nnUNET
# python create_fake_img.py --resume results/mmwhs_run_fold_2/.pth --name mmwhs_run_fold_2 --a2b 1 --data_dir1 ../../../data/MMWHS/CT_withGT_proc/ --cases_folds 2 --result_dir ../../../data/other/fake_MR/ --data_type nnUNET
# python create_fake_img.py --resume results/mmwhs_run_fold_3/.pth --name mmwhs_run_fold_3 --a2b 1 --data_dir1 ../../../data/MMWHS/CT_withGT_proc/ --cases_folds 3 --result_dir ../../../data/other/fake_MR/ --data_type nnUNET
# python create_fake_img.py --resume results/mmwhs_run_fold_4/.pth --name mmwhs_run_fold_4 --a2b 1 --data_dir1 ../../../data/MMWHS/CT_withGT_proc/ --cases_folds 4 --result_dir ../../../data/other/fake_MR/ --data_type nnUNET