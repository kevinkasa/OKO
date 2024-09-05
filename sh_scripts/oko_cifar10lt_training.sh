#!/bin/bash
# Parameters
#SBATCH --cpus-per-task=4
#SBATCH --error=/scratch/ssd004/scratch/kkasa/code/OKO/results/%j_0_log.err
#SBATCH --exclude=gpu179
#SBATCH --gpus-per-node=1
#SBATCH --job-name=OKO
#SBATCH --mem=100GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=/scratch/ssd004/scratch/kkasa/code/OKO/results/%j_0_log.out
#SBATCH --partition=a40
#SBATCH --time 1:00:00
#SBATCH --qos=m2

dataset='cifar10'
#dataset='cifar10_lt'
out_path="/scratch/ssd004/scratch/kkasa/code/OKO/results/${dataset}"
# network='ResNet18';
network='Custom'

n_classes=10
targets='hard'
optim='sgd'
burnin=35
patience=15
steps=40

# sampling_strategies=( 'uniform' 'dynamic' );
sampling_strategies=('uniform')

num_odds=(1 )
max_epochs=(200 )
oko_batch_sizes=(128)
main_batch_sizes=(128)
etas=(0.001 0.001 0.001 0.001 0.001 0.001)
# etas=( 0.01 0.01 0.01 0.01 0.01 0.01 );
num_sets=(30000)
seeds=(0)

source ~/.bashrc
source ~/venvs/oko/bin/activate

export LD_LIBRARY_PATH=/scratch/ssd001/pkgs/cuda-11.3/lib64:/scratch/ssd001/pkgs/cudnn-11.4-v8.2.4.15/lib64:$LD_LIBRARY_PATH
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/scratch/ssd001/pkgs/cuda-11.3
export PATH=/scratch/ssd001/pkgs/cuda-11.3/bin:$PATH

export XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# export XLA_PYTHON_CLIENT_MEM_FRACTION=.7
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

echo "Started odd-k-out learning $SGE_TASK_ID for $network at $(date)"

for sampling in "${sampling_strategies[@]}"; do

  logdir="./logs/${dataset}/${network}/${sampling}/$SGE_TASK_ID"
  mkdir -p $logdir

  python main.py --out_path $out_path --network $network --dataset $dataset --optim $optim --sampling $sampling --n_classes $n_classes --targets $targets --k ${num_odds[@]} --num_sets ${num_sets[@]} --oko_batch_sizes ${oko_batch_sizes[@]} --main_batch_sizes ${main_batch_sizes[@]} --epochs ${max_epochs[@]} --etas ${etas[@]} --burnin $burnin --patience $patience --steps $steps --seeds ${seeds[@]} --regularization >>${logdir}/ooo_${sampling}.out

done

printf "Finished odd-k-out learning $SGE_TASK_ID for $network at $(date)\n"
