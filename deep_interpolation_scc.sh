#!/bin/bash -l

#specify array job
#$ -t 1-62

#this makes it so you'll get an email at the (b)eginning of the job, (e)nd of the job, and on an (a)bort of the job
#$ -m ea

#this merges output and error files into one file
#$ -j y

#this sets the project for the script to be run under
#$ -P jchenlab

#Specify the time limit
#$ -l h_rt=5:00:00

#Specify number of GPUs
#$ -l gpus=1

#Specify GPU Type
#$ -l gpu_type=V100

module load python3
pip install --user s3fs
module load tensorflow/2.3.1


cd /usr3/bustaff/dlamay/deepinterpolation/

python setup.py install

python inference_GPU_SCC.py
