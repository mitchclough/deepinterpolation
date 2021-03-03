#!/bin/bash -l

#this requests a node with a certain number of cpu cores
#$ -pe omp 16
#$ -l gpus=1
#$ -l gpu_type=V100
#$ -l h_rt=24:00:00


#this makes it so you'll get an email at the (b)eginning of the job, (e)nd of the job, and on an (a)bort of the job
#$ -m bea

#this merges output and error files into one file
#$ -j y

#this sets the project for the script to be run under
#$ -P jchenlab


module load python3
pip install --user s3fs
module load tensorflow/2.3.1


cd /usr3/bustaff/dlamay/deepinterpolation/


python setup.py  install

python multi_ophys_training_chen_lab.py
