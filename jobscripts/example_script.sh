#!/bin/bash
#
# Copyright (C) 2023 Cyprien Quéméneur
# For the full license, please refer to the LICENSE file in the root directory of this project.
# For the full copyright notices, please refer to the NOTICE file in the root directory of this project.

#SBATCH --nodes=11                       # total number of nodes (1 server and 10 client nodes)
#SBATCH --gpus-per-node=v100l:1          # total of 11 GPUs
#SBATCH --ntasks-per-gpu=1               # 1 MPI process is launched per node
#SBATCH --cpus-per-task=8                # CPU cores per MPI process
#SBATCH --mem-per-cpu=4G                 # host memory per CPU core (1GB = 1024MB)
#SBATCH --time=0-12:00:00                # time (DD-HH:MM:SS)
#SBATCH --mail-user=myemail@gmail.com    # receive mail notifications
#SBATCH --mail-type=ALL

# Check GPU on orchestrating node
nvidia-smi

# Load modules
module purge
module load python/3.9.6 scipy-stack
module load openmpi/4.0.3
module load gcc/9.3.0
module load opencv/4.6.0
module load mpi4py

# Load pre-existing virtual environment
source ~/venv-py39-fl/bin/activate

# Transmit all files besides the datasets folder to the local storage of the compute nodes
srun rsync -a --exclude="datasets" ../fedpylot $SLURM_TMPDIR

# Create an empty directory on the compute nodes local storage to receive their respective local dataset
srun mkdir -p $SLURM_TMPDIR/fedpylot/datasets/nuimages10

# Transfer the local datasets from the network storage to the local storage of the compute nodes
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun --cpus-per-task=$SLURM_CPUS_PER_TASK python datasets/scatter_data.py --dataset nuimages10

# Move to local storage
cd $SLURM_TMPDIR/fedpylot

# Download pre-trained weights on the orchestrating node (i.e. the server)
if [[ $SLURM_PROCID -eq 0 ]]; then
    bash weights/get_pweights.sh yolov7
fi

# Launch federated learning experiment (see main.py for more details on the settings)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun --cpus-per-task=$SLURM_CPUS_PER_TASK python main.py \
    --nrounds 30 \
    --epochs 5 \
    --server-opt fedavgm \
    --server-lr 1.0 \
    --beta 0.1 \
    --architecture yolov7 \
    --weights weights/pretrained/yolov7_training.pt \
    --data data/nuimages10.yaml \
    --bsz-train 32 \
    --bsz-val 32 \
    --img 640 \
    --conf 0.001 \
    --iou 0.65 \
    --cfg yolov7/cfg/training/yolov7.yaml \
    --hyp data/hyp.scratch.clientopt.nuimages.yaml \
    --workers 8

# Backup experiments folder to network storage
cp -r ./experiments  $myprojects/results