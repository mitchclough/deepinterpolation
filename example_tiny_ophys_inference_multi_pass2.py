import os
from datetime import datetime
import pathlib


def inference(path,start,end):
    import os
    from datetime import datetime
    from deepinterpolation.generic import JsonSaver, ClassLoader
    import numpy as np
    import scipy.io as sio
    
    print("starting inference")
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
    
    generator_param["train_path"] = path

    
    generator_param["batch_size"] = 1
    generator_param["start_frame"] = start
    generator_param["end_frame"] = end # -1 to go until the end.
    generator_param[
        "randomize"
    ] = 0  # This is important to keep the order and avoid the randomization used during training
    
    
    inferrence_param["type"] = "inferrence"
    inferrence_param["name"] = "core_inferrence"
    
    # Replace this path to where you stored your model
    inferrence_param[
        "model_path"
    ] = "/usr3/bustaff/dlamay/deepinterpolation/2021_03_22_13_24_transfer_mean_squared_error_rigid_test_train_bad.h5"
   
    inferrence_param["mat_file"] = path.replace(".mat","_dp.mat")
    
    jobdir = "/usr3/bustaff/dlamay/deepinterpolation/"
    
    try:
        os.mkdir(jobdir)
    except:
        print("folder already exists")
    
    tag=path.split("/")[-1].replace('.mat','')
    
    print("create generators")
    path_generator = os.path.join(jobdir, "generator" + tag + ".json")
    json_obj = JsonSaver(generator_param)
    json_obj.save_json(path_generator)
    
    path_infer = os.path.join(jobdir, "inferrence.json")
    json_obj = JsonSaver(inferrence_param)
    json_obj.save_json(path_infer)
    
    print("build generator")
    generator_obj = ClassLoader(path_generator)
    data_generator = generator_obj.find_and_build()(path_generator)
    
    print("build inference")
    inferrence_obj = ClassLoader(path_infer)
    print("build inference")
    inferrence_class = inferrence_obj.find_and_build()(path_infer, data_generator)
    print("done building")
    
    # Except this to be slow on a laptop without GPU. Inference needs parallelization to be effective.
    
    print("load mat")
    old=loadmat(path.replace(".mat","_dp.mat"))["inference_data"]
    old_id = loadmat(path.replace(".mat","_dp.mat"))["frame_id"]
    new_id = data_generator.list_samples[0:len(data_generator)*5]
    framedata = np.concatenate([np.squeeze(old_id),new_id])
    print("running inference")
    out = inferrence_class.run()
    matdata = np.ascontiguousarray(out)
    matdata = matdata[:,data_generator.a:512-data_generator.a,data_generator.b:512-data_generator.b]
    old = np.ascontiguousarray(np.swapaxes(old, 1, 2))
    old = np.ascontiguousarray(np.swapaxes(old, 0, 1))
    matsavedata=np.concatenate([old,matdata],0)
    matsavedata = np.swapaxes(matsavedata, 0, 2)
    matsavedata = np.swapaxes(matsavedata, 0, 1)
    sio.savemat(path.replace(".mat","_dp.mat"), mdict={'inference_data':matsavedata, 
                                                        'frame_id':framedata})
    
    
    os.remove(path_generator)
    os.remove(path_infer)

    





from tqdm import tqdm
import sys
from scipy.io import loadmat
import numpy as np

startTime=datetime.now()


for sess in tqdm(range(1,2)):
    local_train_path ='/net/claustrum2/mnt/data/Projects/Perirhinal/Animals/pr012/2P/pr012-' + str(sess) + '/PreProcess/A1_Ch0'
    #local_train_path = os.path.join(os.environ['TMPDIR'],'A0_Ch0')
    import glob
    train_paths = sorted(set(glob.glob(os.path.join(local_train_path,'*.mat')))-set(glob.glob(os.path.join(local_train_path,'*_dp.mat'))))
    train_paths_done = glob.glob(os.path.join(local_train_path,'*_dp.mat'))
   
    paths_td=[]
    for i in tqdm(train_paths):
        if i.replace('.mat','_dp.mat') in train_paths_done:
            mat_file = loadmat(i)['motion_corrected']
            dp_file= loadmat(i.replace('.mat','_dp.mat'))['inference_data']
            start=int(np.floor(float(mat_file.shape[2]-60)) / 5)*5 #to grab extra frames missed by batch size
            end = mat_file.shape[2]-1
            if dp_file.shape[2] != mat_file.shape[2]-60:
                print(i)
                inference(i,start,end)
    
    print(datetime.now() - startTime)
            
print(datetime.now() - startTime)
    
    #try:
    
    #except:
        #print("corrupt file")
    

    
