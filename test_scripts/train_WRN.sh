#!/bin/bash	

#SBATCH --job-name=PR_Train_MNIST
#SBATCH --output=PR_Train_MNIST-%J.out
#SBATCH --mail-user=mikkjo@dtu.dk
#SBATCH --mail-type=END,FAIL
#SBATCH --export=ALL
#SBATCH --gres=gpu
#SBATCH --mem=20g

cd /home/mikkjo/projects/TST_code/
export PYTHONPATH=$PWD
source activate WriteReader
echo "!!Training model!!"
srun python3 src/experiments/00_train_models.py \
    --model WRN \
    --epochs 1 \
    --accelerator gpu \
    --seed 1 \
    --dataset CIFAR10 \
    --model_name CIFAR10_WRN_28_10_Base \
    --batch_size 256
echo "!!Training done!!"
