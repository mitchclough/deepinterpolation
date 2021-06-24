

import os
import sys
import glob
import json

#change to animal name
animal = 'pr012'
#change to animal path
animal_path ='/net/claustrum2/mnt/data/Projects/Perirhinal/Animals/' + animal + '/2P/'
local_train_paths = glob.glob(os.path.join(animal_path, animal +'-*/PreProcess/*_Ch0'))

train_paths_td = []
for local_train_path in local_train_paths:
    train_paths = sorted(set(glob.glob(os.path.join(local_train_path,'*.mat')))-set(glob.glob(os.path.join(local_train_path,'*_dp.mat'))))
    train_paths_done = glob.glob(os.path.join(local_train_path,'*_dp.mat'))
    for i in train_paths:
        if i.replace('.mat','_dp.mat') not in train_paths_done:
            train_paths_td.append(i)
print(len(train_paths_td))

json_obj = json.dumps(train_paths_td)

#change to name json file
with open("pr012_files.json", "w") as outfile:
    outfile.write(json_obj)
