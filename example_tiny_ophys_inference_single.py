import os
from deepinterpolation.generic import JsonSaver, ClassLoader
import pathlib
from datetime import datetime
import numpy as np
import scipy.io as sio
from scipy.io import loadmat


startTime=datetime.now()

generator_param = {}
inferrence_param = {}

# We are reusing the data generator for training here.
generator_param["type"] = "generator"
generator_param["name"] = "SingleTifGenerator"
generator_param["pre_post_frame"] = 30
generator_param["pre_post_omission"] = 0
generator_param[
    "steps_per_epoch"
] = -1  # No steps necessary for inference as epochs are not relevant. -1 deactivate it.

generator_param["train_path"] = "X:/Projects/Perirhinal/Animals/pr020/2P/pr020-1/PreProcess/A1_Ch0/A1_Ch0_15-31-17.mat"
#os.path.join(
    #pathlib.Path(__file__).parent.absolute(),
    #"..",
    #"sample_data",
    #"A0_Ch0_16-44-59.mat",
#)

mat_file = loadmat(generator_param["train_path"])['motion_corrected']
start=int(np.floor(float(mat_file.shape[2]-60)) / 5)*5   #to grab extra frames missed by batch size

generator_param["batch_size"] = 1
generator_param["start_frame"] = start
generator_param["end_frame"] = mat_file.shape[2]-1  # -1 to go until the end.
generator_param[
    "randomize"
] = 0  # This is important to keep the order and avoid the randomization used during training


inferrence_param["type"] = "inferrence"
inferrence_param["name"] = "core_inferrence"

# Replace this path to where you stored your model
inferrence_param[
    "model_path"
] = "X:/Projects/Perirhinal/deepinterpolation/trained_models/Training_models/2021_03_22_13_24_transfer_mean_squared_error_rigid_test_train_bad.h5"

# Replace this path to where you want to store your output file
#inferrence_param[
    #"output_file"
#] = "X:/Projects/Perirhinal/deepinterpolation/trained_models/ophys_tiny_continuous_deep_interpolation_pr020-28_A1_Ch0_10-16-03_transfer_rigid_test_train_bad.h5"

inferrence_param["mat_file"] = generator_param["train_path"].replace(".mat","_dp.mat")

jobdir = "X:/Projects/Perirhinal/deepinterpolation/trained_models/"

try:
    os.mkdir(jobdir)
except:
    print("folder already exists")

path_generator = os.path.join(jobdir, "generator.json")
json_obj = JsonSaver(generator_param)
json_obj.save_json(path_generator)

path_infer = os.path.join(jobdir, "inferrence.json")
json_obj = JsonSaver(inferrence_param)
json_obj.save_json(path_infer)

generator_obj = ClassLoader(path_generator)
data_generator = generator_obj.find_and_build()(path_generator)



inferrence_obj = ClassLoader(path_infer)
inferrence_class = inferrence_obj.find_and_build()(path_infer, data_generator)


# Except this to be slow on a laptop without GPU. Inference needs parallelization to be effective.

path = generator_param["train_path"]

old=loadmat(path.replace(".mat","_dp.mat"))["inference_data"]
old_idx=loadmat(path.replace(".mat","_dp.mat"))["frame_id"]
out = inferrence_class.run()
matdata = np.ascontiguousarray(out)
matdata = matdata[:,data_generator.a:512-data_generator.a,data_generator.b:512-data_generator.b]
old = np.ascontiguousarray(np.swapaxes(old, 1, 2))
old = np.ascontiguousarray(np.swapaxes(old, 0, 1))
matsavedata=np.concatenate([old,matdata],0)
matsavedata = np.swapaxes(matsavedata, 0, 2)
matsavedata = np.swapaxes(matsavedata, 0, 1)
print(data_generator.list_samples)
sio.savemat(path.replace(".mat","_dp.mat"), mdict={'inference_data':matsavedata})

os.remove(path_generator)
os.remove(path_infer)

print(datetime.now() - startTime)
