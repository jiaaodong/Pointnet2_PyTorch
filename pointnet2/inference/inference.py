import sys 
import os.path as osp 
import os 
import numpy as np 


from pointnet2.models import Pointnet2SemMSG as Pointnet
import torch 
import torch.nn 
import h5py

model_path = "/home/jiaaodong/Pointnet2_PyTorch/pointnet2/train/checkpoints/poitnet2_semseg_best.pth.tar"
data_path = "/home/jiaaodong/Pointnet2_PyTorch/pointnet2/data/RadarLowLevel/radar_pcd_dataset_multi_baseline.h5"

model = Pointnet(num_classes=4, input_channels=2, use_xyz=True)
# model.load_state_dict(torch.load(model_path))
model.eval()
model.cuda()

if os.path.isfile(model_path):
    print("==> Loading from checkpoint '{}'".format(model_path))
    checkpoint = torch.load(model_path)
    epoch = checkpoint["epoch"]
    it = checkpoint.get("it", 0.0)
    best_prec = checkpoint["best_prec"]
    if model is not None and checkpoint["model_state"] is not None:
        model.load_state_dict(checkpoint["model_state"])
    print("==> Done")



data_file  = h5py.File(data_path)
features = data_file["data"][:]
labels = data_file["label"][:]
print("shape of the features:{} and labels:{}".format(features.shape, labels.shape))