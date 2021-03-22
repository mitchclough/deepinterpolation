import tensorflow as tf
# Allow soft placement so your job runs on the assigned GPU.
tf.config.set_soft_device_placement(True)

import deepinterpolation as de
import sys
from shutil import copyfile
import os
from deepinterpolation.generic import JsonSaver, ClassLoader
import datetime
from typing import Any, Dict
import glob
import csv

now = datetime.datetime.now()
run_uid = now.strftime("%Y_%m_%d_%H_%M")

training_param = {}
generator_param = {}
network_param = {}
generator_test_param = {}

steps_per_epoch = 10

generator_test_param["type"] = "generator"
generator_test_param["name"] = "SingleTifGenerator"
generator_test_param["pre_post_frame"] = 30
generator_test_param['pre_post_omission'] = 1

generator_test_param[
    "train_path"
] = "/net/claustrum2/mnt/data/Projects/Perirhinal/Animals/pr012/2P/pr012-32/PreProcess/A1_Ch0/A1_Ch0_10-56-09.mat"
generator_test_param["batch_size"] = 5
generator_test_param["start_frame"] = 0
generator_test_param["end_frame"] = 639
generator_test_param["steps_per_epoch"] = -1
generator_test_param["randomize"] = 0

#local_train_path = '/net/claustrum2/mnt/data/Projects/Perirhinal/Animals/pr012/2P/pr012-1/PreProcess/A0_Ch0'
#local_train_path = os.path.join(os.environ['TMPDIR'],'A0_Ch0')

#train_paths = glob.glob(os.path.join(local_train_path,'*.mat'))

# Use the next 3 lines to add different sessions/animals
#local_train_paths = []

train_paths = []
frame_nums = []

with open('/net/claustrum2/mnt/data/Projects/Perirhinal/deepinterpolation/train_paths_80_bad.csv','r') as csv_file:
    for a in csv.reader(csv_file, delimiter=','):
        train_paths.append(a[0])
        frame_nums.append(int(a[1]))

#for local_train_path in local_train_paths:
    #train_paths.extend([f for f in glob.glob(os.path.join(local_train_path,'*.mat')))

generator_param_list = []
for i in range(len(train_paths)):

    generator_param = {}

    generator_param["type"] = "generator"
    generator_param["name"] = "SingleTifGenerator"
    generator_param["pre_post_frame"] = 30
    generator_param["train_path"] = train_paths[i]
    generator_param["batch_size"] = 5
    generator_param["start_frame"] = 0
    generator_param["end_frame"] = frame_nums[i]
    generator_param["steps_per_epoch"] = steps_per_epoch
    generator_param["randomize"] = 1
    generator_param["pre_post_omission"] = 1

    generator_param_list.append(generator_param)

training_param["type"] = "trainer"
training_param["name"] = "transfer_trainer"
#Change this path to any model you wish to improve
training_param[
    "model_path"
] = r"/usr3/bustaff/dlamay/deepinterpolation/2019_09_11_23_32_unet_single_1024_mean_absolute_error_Ai148-0450.h5"

training_param["run_uid"] = run_uid
training_param["batch_size"] = generator_test_param["batch_size"]
training_param["steps_per_epoch"] = steps_per_epoch
training_param["period_save"] = 25
training_param["nb_gpus"] = 2
training_param["apply_learning_decay"] = 0
training_param["pre_post_frame"] = generator_test_param["pre_post_frame"]
training_param["nb_times_through_data"] = 1
training_param["learning_rate"] = 0.0005
training_param["loss"] = "mean_squared_error"   #change to mean_squareroot_error for L2 low SNR
training_param[
    "nb_workers"
] = 16
training_param["caching_validation"]=False

training_param["model_string"] = (
    "transfer"
    + "_"
    + training_param["loss"]
    + "_"
    + training_param["run_uid"]
)

jobdir = (
    "/projectnb/jchenlab/trained_models/"
    + training_param["model_string"]
    + "_"
    + run_uid
)

training_param["output_dir"] = jobdir

try:
    os.mkdir(jobdir, 0o775)
except:
    print("folder already exists")


path_training = os.path.join(jobdir, "training.json")
json_obj = JsonSaver(training_param)
json_obj.save_json(path_training)

list_train_generator = []
for local_index, indiv_generator in enumerate(generator_param_list):


    path_generator = os.path.join(jobdir, "generator" + str(local_index) + ".json")
    json_obj = JsonSaver(indiv_generator)
    json_obj.save_json(path_generator)
    generator_obj = ClassLoader(path_generator)
    train_generator = generator_obj.find_and_build()(path_generator)



    list_train_generator.append(train_generator)


path_test_generator = os.path.join(jobdir, "test_generator.json")
json_obj = JsonSaver(generator_test_param)
json_obj.save_json(path_test_generator)

#path_network = os.path.join(jobdir, "network.json")
#json_obj = JsonSaver(network_param)
#json_obj.save_json(path_network)

generator_obj = ClassLoader(path_generator)
generator_test_obj = ClassLoader(path_test_generator)

#network_obj = ClassLoader(path_network)
trainer_obj = ClassLoader(path_training)

train_generator = generator_obj.find_and_build()(path_generator)

global_train_generator = de.generator_collection.CollectorGenerator(
    list_train_generator
)

test_generator = generator_test_obj.find_and_build()(path_test_generator)

#network_callback = network_obj.find_and_build()(path_network)

training_class = trainer_obj.find_and_build()(
    global_train_generator, test_generator, path_training
)

training_class.run()

training_class.finalize()
