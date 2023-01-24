#!/bin/bash -l

#specify array job
#superseded by submit script
# -t 1-500

#this makes it so you'll get an email at the (b)eginning of the job, (e)nd of the job, and on an (a)bort of the job
#$ -m ea

#this merges output and error files into one file
#$ -j y

#this sets the project for the script to be run under
#$ -P jchenlab

#$ -pe omp 16

module load miniconda
conda activate deepinterpolation
module load tensorflow

python inference_CPU_SCC.py
