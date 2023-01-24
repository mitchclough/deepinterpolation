#!/usr/bin/bash -l

anm=$1
sess=$2
chan=1

module load python3
cd ~/deepinterpolation
python create_json.py "$anm" "$sess" "$chan"
module unload python3

length=$(< "$anm"-"$sess".length)

cd ~
jobs=$(( length / 100 + 1 ))

qsub -N "$anm"-"$sess" -t 1-"$jobs" deep_interpolation_scc.sh "$anm" "$sess"