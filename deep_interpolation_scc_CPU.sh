#!/bin/bash -l

#specify array job
#$ -t 1-500


#this makes it so you'll get an email at the (b)eginning of the job, (e)nd of the job, and on an (a)bort of the job
#$ -m ea

#this merges output and error files into one file
#$ -j y

#this sets the project for the script to be run under
#$ -P jchenlab

#$ -pe omp 16


#Specify the time limit
#$ -l h_rt=2:00:00



module load python3/3.7.9
pip install --user s3fs
module load tensorflow/2.3.1

python inference_CPU_SCC.py
