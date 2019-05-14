import sys 
import os.path as osp 
import os 
import argparse
import numpy as np 


from pointnet2.models import Pointnet2SemMSG as Pointnet
import torch 
import torch.nn 
import h5py
import matplotlib.pyplot as plt 
import time

from copy import deepcopy

home = osp.expanduser("~")
top_folder = osp.join(osp.dirname(os.path.abspath(__file__)), "..")
model_path = osp.join(top_folder, 'train','checkpoints','poitnet2_semseg_best.pth.tar')
data_path = osp.join(top_folder, "data","RadarLowLevel","radar_pcd_dataset_multi_baseline.h5")

parser = argparse.ArgumentParser()
parser.add_argument("-model", type = str, default = model_path)
parser.add_argument("-data", type= str, default = data_path)
parser.add_argument("-root", type= str, default = osp.join("/data", 'jiaaodong','conti_new','dataset_daimler'))
args = parser.parse_args()

def load_pointnet(model_path):
    model = Pointnet(num_classes=4, input_channels=2, use_xyz=True)
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
    return model 

def load_data(data_path):
    data_file  = h5py.File(data_path)
    features = data_file["data"][:]
    labels = data_file["label"][:]
    return features, labels 


if __name__ == "__main__":
    
    data_path = args.data
    model_path = args.model
    root_path = args.root
    img_path = osp.join(root_path, "pointnet_result")
    if not osp.exists(img_path):
        print("Creating image folder...")
        os.makedirs(img_path)
    features, labels = load_data(data_path)
    points_all = deepcopy(features[:, :, :3])
    model = load_pointnet(model_path)
    #print("shape of the features:{} and labels:{} and pred:{}, inference time:{}".format(features.shape, labels.shape, pred.shape,inference_time))

    for i, points_single in enumerate(points_all):
        features_single = torch.tensor(np.expand_dims(features[i,...],axis=0)).type(torch.FloatTensor)
        t1 = time.time()
        pred = model(features_single.to("cuda", non_blocking=True))
        _, pred = torch.max(pred, -1)
        pred = pred.cpu().detach().numpy()
        t2 = time.time()
        inference_time = t2-t1 
        fig = plt.figure(figsize=(16,8))
        ax1 = plt.subplot(121)
        sc1 = ax1.scatter(points_single[:,1], points_single[:,0], c=labels[i, :])
        ax1.set_ylim([0,100])
        ax1.set_xlim([-30,30])
        ax1.set_title("Input")
        plt.colorbar(sc1)
        ax2 = plt.subplot(122)
        sc2=ax2.scatter(points_single[:,1], points_single[:,0], c=pred[0,:])
        ax2.set_ylim([0,100])
        ax2.set_xlim([-30,30])
        ax2.set_title("Prediction")
        plt.colorbar(sc2)
        plt.savefig(osp.join(img_path,"{:05d}.jpg".format(i)))
        plt.close()
        print("finished {}th frame in {} inference time".format(i, inference_time))
        # plt.show()

