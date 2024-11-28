#!/bin/bash
#SBATCH --job-name=test_DRIT_MMWHS                       # Job name
#SBATCH --partition=rtx2080ti                       # Partition
#SBATCH --gpus-per-task=1                           # Number of gpus per node
#SBATCH --gpus=1                                    # Number of gpus in total
#SBATCH --ntasks=1                                  # Run on a single node
#SBATCH --cpus-per-task=4                           # Number of cores
#SBATCH --time=01:00:00                             # Time limit hrs:min:sec
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

# Training
cd src
rsync -avv --info=progress2 --delete /projects/multimodal_disentanglement_review/nnUNet/nnUNet_preprocessed/Dataset033_MMWHSCT $SCRATCH/data_dir1
rsync -avv --info=progress2 --delete /projects/multimodal_disentanglement_review/nnUNet/nnUNet_preprocessed/Dataset034_MMWHSMRI $SCRATCH/data_dir2
# Train
python train.py --name mmwhs_run_fold_0 --batch_size 188 --data_dir1 $SCRATCH/data_dir1/Dataset033_MMWHSCT --data_dir2 $SCRATCH/data_dir2/Dataset034_MMWHSMRI --cases_folds 0 --data_type nnUNet --n_ep 1000 --splits_dir1 ../splits/splits_mmwhs_ct.json --splits_dir2 ../splits/splits_mmwhs_mri.json

# python evaluate_model.py --name mmwhs_run_fold_0 --data_type nnUNet --cases_fold 0 --data_dir1 $SCRATCH/data_dir1/Dataset_033_MMWHSCT/ --data_dir2 $SCRATCH/data_dir2/Dataset_034_MMWHSMRI/
# python create_fake_img.py --resume results/mmwhs_run_fold_0/.pth --name mmwhs_run_fold_0 --a2b 0 --data_dir1 $SCRATCH/data_dir2/Dataset034_MMWHSMRI/ --cases_folds 0 --result_dir ../synthetic_data/Dataset041_MMWHSFAKECT/ --name_new_ds MMWHSFAKECT --data_type nnUNet
# python create_fake_img.py --resume results/mmwhs_run_fold_0/.pth --name mmwhs_run_fold_0 --a2b 1 --data_dir1 $SCRATCH/data_dir1/Dataset033_MMWHSCT/ --cases_folds 0 --result_dir ../synthetic_data/Dataset042_MMWHSFAKEMRI/ --name_new_ds MMWHSFAKEMRI --data_type nnUNet