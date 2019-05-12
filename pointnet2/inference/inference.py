import sys 
import os.path as osp 
import os 
import numpy as np 


from pointnet2.models import Pointnet2SemMSG as Pointnet
import torch 
import torch.nn 
import h5py
import matplotlib.pyplot as plt 
import time

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


indx = 10
points = features[indx,:,:3]
labels = np.expand_dims(labels[indx], axis=0)
sample = features[indx,:]
sample = torch.tensor(np.expand_dims(sample,axis=0)).type(
            torch.FloatTensor
        )

t1 = time.time()
pred = model(sample.to("cuda", non_blocking=True))
_, pred = torch.max(pred, -1)
pred = pred.cpu().detach().numpy()
t2 = time.time()
inference_time = t2-t1 
print("shape of the features:{} and labels:{} and pred:{}, infer time:{}".format(features.shape, labels.shape, pred.shape,inference_time))
ax1 = plt.subplot(121)

ax1.scatter(points[:,1], points[:,0], c=labels[0,:])
ax2 = plt.subplot(122)
ax2.scatter(points[:,1], points[:,0], c=pred[0,:])
plt.show()
