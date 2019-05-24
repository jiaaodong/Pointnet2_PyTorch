import sys 
import os.path as osp 
import os 
import argparse
import numpy as np 


from pointnet2.models import Pointnet2SemMSG as Pointnet
import torch 
import torch.nn as nn 
import h5py
import matplotlib.pyplot as plt 
import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from plot_confusion_matrix import plot_confusion_matrix

from torch.utils.data import DataLoader
import etw_pytorch_utils as pt_utils
from pointnet2.data import RadarLowLvlSemSeg

import tqdm

from copy import deepcopy
from infer_utils import Tester

home = osp.expanduser("~")
top_folder = osp.join(osp.dirname(os.path.abspath(__file__)), "..")
model_path = osp.join(top_folder, 'train','checkpoints','poitnet2_semseg_best.pth.tar')

parser = argparse.ArgumentParser()
parser.add_argument("-model", type = str, default = model_path)
parser.add_argument("-root", type= str, default = osp.join("/data", 'jiaaodong','conti_new','dataset1'))
parser.add_argument("-data", type= str, default = "/home/jiaaodong/Pointnet2_PyTorch/pointnet2/data/RadarLowLevel/LLT_without_static.h5")
args = parser.parse_args()

def load_pointnet(model_path):
    model = Pointnet(num_classes=4, input_channels=72, use_xyz=False)
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
    # features = np.insert(features, 2, 0, axis=2)
    return features, labels 




if __name__ == "__main__":
    use_gpu = True
    model_path = args.model
    root_path = args.root
    data_path = args.data
    num_pts = 128
    draw_plots = True
    result_folder = "rm_static_nearby"
    result_folder = osp.join(root_path, result_folder)
    if not osp.exists(result_folder):
        os.makedirs(result_folder)

    train_set = RadarLowLvlSemSeg(num_pts, train_rat=0.6)
    train_loader = DataLoader(
        train_set,
        batch_size=1024,
        pin_memory=True,
        num_workers=2,
        shuffle=False,
    )

    test_set = RadarLowLvlSemSeg(num_pts, train=False)
    test_loader = DataLoader(
        test_set,
        batch_size=1024,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )

    model = load_pointnet(model_path)
    weights = [0.00312226, 0.51293818, 0.42739984, 0.05653972]
    weights = torch.tensor(weights)
    if use_gpu:
        weights = weights.cuda().float()
    loss_func = nn.CrossEntropyLoss(weights)
    tester = Tester(
        model,
        loss_func = loss_func,
        use_gpu=use_gpu
    )

    loss_train, pred_labels_train, true_labels_train = tester.test(
        train_loader
    )
    print("loss train:{}".format(loss_train))

    loss_val, pred_labels_val, true_labels_val = tester.test(
        test_loader
    )
    print("loss val:{}".format(loss_val))

    np.save(osp.join(result_folder, "pred_labels_train_pnt2"), pred_labels_train)
    np.save(osp.join(result_folder, "true_labels_train_pnt2"), true_labels_train)
    np.save(osp.join(result_folder, "pred_labels_val_pnt2"), pred_labels_val)
    np.save(osp.join(result_folder, "true_labels_val_pnt2"), true_labels_val)

    class_names = np.array(['background', 'pedestrian', 'rider', 'car'])
    confusion_mat = confusion_matrix(true_labels_val.flatten(), pred_labels_val.flatten())
    f1_score_val = f1_score(true_labels_val, pred_labels_val)
    fig, ax1=plt.subplots(figsize=(8, 7))
    plot_confusion_matrix(confusion_mat, classes=class_names,
                title='pnt2_without_normalization_val', ax = ax1)
    plt.savefig(osp.join(result_folder, 'pnt2_unnormalized_val'))

    fig, ax1=plt.subplots(figsize=(8, 7))
    plot_confusion_matrix(confusion_mat, classes=class_names,normalize=True,
                title='pnt2_normalization_val', ax = ax1)
    plt.savefig(osp.join(result_folder,  'pnt2_normalized_val'))
    np.savetxt(osp.join(result_folder, 'pnt2_val.txt'),confusion_mat,fmt='%.2f')
    np.savetxt(osp.join(result_folder, 'f1score_val.txt'),f1_score_val,fmt='%.2f')
    np.save(osp.join(result_folder, 'pnt2_conmat_val.npy'),confusion_mat)

    confusion_mat = confusion_matrix(true_labels_train.flatten(), pred_labels_train.flatten())
    f1_score_val = f1_score(true_labels_train, pred_labels_train)
    fig, ax1=plt.subplots(figsize=(8, 7))
    plot_confusion_matrix(confusion_mat, classes=class_names,
                title='pnt2_without_normalization_train', ax = ax1)
    plt.savefig(osp.join(result_folder, 'pnt2_unnormalized_train'))

    fig, ax1=plt.subplots(figsize=(8, 7))
    plot_confusion_matrix(confusion_mat, classes=class_names,normalize=True,
                title='pnt2_normalization_train', ax = ax1)
    plt.savefig(osp.join(result_folder,  'pnt2_normalized_train'))
    np.savetxt(osp.join(result_folder, 'pnt2_train.txt'),confusion_mat,fmt='%.2f')
    np.savetxt(osp.join(result_folder, 'f1score_train.txt'),f1_score_val,fmt='%.2f')
    np.save(osp.join(result_folder, 'pnt2_conmat_train.npy'),confusion_mat)

