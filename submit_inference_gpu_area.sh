#!/usr/bin/bash -l

anm=$1
area=$2
chan=1

module load python3
cd ~/deepinterpolation
python create_json_area.py "$anm" "$area" "$chan"
module unload python3

length=$(< "$anm"-A"$area".length)
jobs=$(( length / 100 + 1 ))

cd ~
qsub -N "$anm"-deep-A"$area" -t 1-"$jobs" deep_interpolation_scc_gpu_area.sh "$anm" "$area"
