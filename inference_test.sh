#!/bin/bash -l

#this requests a node with a certain number of cpu cores
#$ -pe omp 16



#this makes it so you'll get an email at the (b)eginning of the job, (e)nd of the job, and on an (a)bort of the job
#$ -m bea

#this merges output and error files into one file
#$ -j y

#this sets the project for the script to be run under
#$ -P jchenlab

module load python3
pip install --user s3fs
module load tensorflow/2.0.0


cd /usr3/bustaff/dlamay/deepinterpolation/examples/


python example_tiny_inference.py
