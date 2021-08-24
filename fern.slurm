#!/bin/bash

#---------------------------------------------------#
#@Author: Fredy
#@Email:  zhenkaiz@student.unimelb.edu.au
#@Create: 16:15 17/02/2021
#@Version: v0.0
#---------------------------------------------------#

#---------------------------------------------------#
#   Config the Slurm                                #
#   Set partition, cpu, GPU....                     #
#---------------------------------------------------#
#SBATCH --job-name="gpu_nerf"
#SBATCH --partition=gpgpu
#SBATCH -q gpgpuresplat
#SBATCH --nodes=1                   # set the node number
#SBATCH --time=3-12:00:00           # set the running time
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --account="punim1006"
#SBATCH --output="job-%N-%J.log"
#SBATCH --error="job-%N-%J.err"

#SBATCH --mail-user=zhenkaiz@student.unimelb.edu.au
# SBATCH --mail-type=FAIL            # aborts abnormally (fails)
# SBATCH --mail-type=BEGIN           # begins
#SBATCH --mail-type=END             # ends successfully

#---------------------------------------------------#
#    Import the necessary modules                   #
#---------------------------------------------------#
module add gcc/8.3.0
module add cuda/10.1.243            # import the module, notes: need import right version of GPU
# module load gcccore/8.3.0
# module load python/3.7.4
source activate nerf_pl_z
# pip install torchsearchsorted/.

#---------------------------------------------------#
#    Running Code                                   #
#---------------------------------------------------#
echo "Starting the process--------------------------------------"

python train.py \
   --dataset_name llff \
   --root_dir "/data/gpfs/projects/punim1006/zzk/dataset/nerf_llff_data/fern" \
   --N_importance 64 --img_wh "504" "378" \
   --num_epochs "10000" --batch_size 3024 \
   --optimizer adam --lr 5e-4 \
   --lr_scheduler cosine \
   --exp_name "fern" \
   --num_gpus 4 \
   --chunk 3024

echo "Finishing the training-------------------------------------"
