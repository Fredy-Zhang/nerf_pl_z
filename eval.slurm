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
#SBATCH --job-name="gpu_nerf_case0_eval"
#SBATCH --partition=gpgpu
#SBATCH -q gpgpuresplat
#SBATCH --nodes=1                   # set the node number
#SBATCH --time=0-12:00:00           # set the running time
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --account="punim1006"
#SBATCH --output="job-%N-%J.log"
#SBATCH --error="job-%N-%J.err"

#SBATCH --mail-user=zhenkaiz@student.unimelb.edu.au
#SBATCH --mail-type=FAIL            # aborts abnormally (fails)
#SBATCH --mail-type=BEGIN           # begins
#SBATCH --mail-type=END             # ends successfully

#---------------------------------------------------#
#    Import the necessary modules                   #
#---------------------------------------------------#
module add cuda/10.1.243            # import the module, notes: need import right version of GPU
# module load gcccore/8.3.0
# module load python/3.7.4
source activate nerf_pl
# pip install torchsearchsorted/.

#---------------------------------------------------#
#    Running Code                                   #
#---------------------------------------------------#
echo "Starting the process--------------------------------------"

#python eval.py \
#   --root_dir "../../dataset/nerf_llff_data/fern" \
#   --dataset_name llff \
#   --scene_name "fern_1" \
#   --img_wh "504" "378" \
#   --N_importance 64 \
#   --ckpt_path "./ckpts/fern_64_relu/epoch=8.ckpt" \
#   --split "test" \
#   --N_samples 64 \
#   --chunk 2048

python eval.py \
   --root_dir "../../dataset/nerf_llff_data/fern" \
   --dataset_name llff \
   --scene_name "fern_case0" \
   --img_wh "504" "378" \
   --N_importance 64 \
   --ckpt_path "./ckpts/fern_case0/epoch=29.ckpt" \
   --split "test" \
   --N_samples 64 \
   --chunk 2048

echo "Finishing the training-------------------------------------"
