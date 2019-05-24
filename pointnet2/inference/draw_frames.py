
import os 
import os.path as osp 
import sys 
import argparse

import matplotlib.pyplot as plt 
import numpy as np 
import tqdm
import h5py

from inference import load_data

parser = argparse.ArgumentParser(description="Parse the arguments to visualize inference")    
parser.add_argument("-root", type= str, default = osp.join("/data", 'jiaaodong','conti_new','dataset1'))
parser.add_argument("-data", type= str, default = "/home/jiaaodong/Pointnet2_PyTorch/pointnet2/data/RadarLowLevel/LLT_without_static.h5")

args = parser.parse_args()
data_path = args.data 
root_path = args.root
result_path = "rm_static_nearby"
result_path = osp.join(root_path, result_path)
if not osp.exists(result_path):
    os.makedirs(result_path)

if __name__ == "__main__":
    pred_labels_train = np.load(osp.join(result_path, "pred_labels_train_pnt2.npy"))
    pred_labels_val   = np.load(osp.join(result_path, "pred_labels_val_pnt2.npy"  ))
    true_labels_train = np.load(osp.join(result_path, "true_labels_train_pnt2.npy"))
    true_labels_val   = np.load(osp.join(result_path, "true_labels_val_pnt2.npy"  ))

    features, labels = load_data(data_path)
    features = features[:, :, :3]
    img_path = osp.join(result_path, "pointnet_result")
    color_map = ['y','g','r','b']
    if not osp.exists(img_path):
        print("Creating image folder...")
        os.makedirs(img_path)

    pred_labels = np.concatenate((pred_labels_train, pred_labels_val), axis=0)
    print(pred_labels.shape)
    true_labels = np.concatenate((true_labels_train, true_labels_val), axis=0)
    print(true_labels.shape)


    pbar = tqdm.tqdm(total = features.shape[0])
    for i in range(features.shape[0]):
        pbar.update()
        fig = plt.figure(figsize=(16,8))
        ax1 = plt.subplot(121)
        sc1 = ax1.scatter(features[i,:,1], features[i,:,0], c=[color_map[int(cindx)] for cindx in labels[i, :]])
        ax1.set_ylim([0,100])
        ax1.set_xlim([-30,30])
        ax1.set_title("Input")
        # plt.colorbar(sc1)
        ax2 = plt.subplot(122)
        sc2=ax2.scatter(features[i,:,1], features[i, :,0], c= [color_map[int(cindx)] for cindx in pred_labels[i, :]])
        ax2.set_ylim([0,100])
        ax2.set_xlim([-30,30])
        ax2.set_title("Prediction")
        # plt.colorbar(sc2)
        plt.savefig(osp.join(img_path,"{:05d}.jpg".format(i)))
        plt.close()
        # plt.show()
    pbar.close()