from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.nn as nn
import etw_pytorch_utils as pt_utils
from collections import namedtuple

from pointnet2.utils.pointnet2_modules_radar import PointnetFPModule, PointnetSAModuleMSG


def model_fn_decorator(criterion):
    ModelReturn = namedtuple("ModelReturn", ["preds", "loss", "acc"])

    def model_fn(model, data, epoch=0, eval=False):
        with torch.set_grad_enabled(not eval):
            inputs, labels = data
            inputs = inputs.to("cuda", non_blocking=True)
            labels = labels.to("cuda", non_blocking=True)

            preds = model(inputs)
            loss = criterion(preds.view(labels.numel(), -1), labels.view(-1))

            _, classes = torch.max(preds, -1)
            acc = (classes == labels).float().sum() / labels.numel()
            True_labels = labels[classes==labels]
            
            # False_labels_GT = labels[classes!=labels]        
            # False_labels_Pred = classes[classes!=labels]
            # num_others = labels[labels==0].size[0]
            # num_ped = (labels[labels==1]).float().sum()
            # num_biker = (labels[labels==2]).size[0]
            # num_car = (labels[labels==3]).size[0]
            # # print("num_car:{}  num_others:{}  num_bikers{}  num_ped:{}".format(num_car, num_others, num_biker, num_ped))
            # acc_T_ped_F_others = (False_labels_Pred[False_labels_GT==1]).numel() / num_ped
            acc_TP_pedestrian = (True_labels==1).float().sum() / (labels==1).float().sum()
            # print(acc_TP_pedestrian)
            # # acc_T_ped_F_biker = (False_labels_Pred[False_labels_GT==1]).numel() / num_ped
            # # acc_T_ped_F_car = (False_labels_Pred[False_labels_GT==2]).numel() /num_ped


            # acc_TP_biker = (True_labels==2).float().sum() / (labels==2).float().sum()
            # acc_TP_car = (True_labels==3).float().sum() / (labels==3).float().sum()
            # acc_TP_others = (True_labels==0).float().sum() / (labels==0).float().sum()
            # print( (True_labels==0).float().sum(), num_others,acc_TP_others)

        return ModelReturn(preds, loss, 
                                {
                                # "Number of inputs": labels.numel(),
                                # "Number of other targets":num_others,
                                # "Sum of labels": labels.float().sum().item(),
                                # "True Positive of others": acc_TP_others.item(),
                                "Number of pedestrian targets":(labels==1).float().sum().item(),
                                "True Positive of pedestrian": acc_TP_pedestrian.item(), 
                                # "Number of biker targets": num_biker.item(),
                                # "True Positive of biker": acc_TP_biker.item(),
                                # "Number of car targets": num_car.item(),
                                # "True Positive of car": acc_TP_car.item(),
                                "loss": loss.item(),
                                "acc": acc.item()})

    return model_fn


class Pointnet2MSG(nn.Module):
    ########## This is what we need to use ######################################################
    r"""
        PointNet2 with multi-scale grouping
        Semantic segmentation network that uses feature propogation layers

        Parameters
        ----------
        num_classes: int
            Number of semantics classes to predict over -- size of softmax classifier that run for each point
        input_channels: int = 6
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, num_classes, input_channels=37, use_xyz=True):
        super(Pointnet2MSG, self).__init__()
        self.scaling_factor = 4
        self.SA_modules = nn.ModuleList()
        c_in = input_channels
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=int(512/self.scaling_factor), # The number of groups
                radii=[1, 3],
                nsamples=[int(8/self.scaling_factor), int(32/self.scaling_factor)], # The number of samples in each group 
                mlps=[[c_in, int(32/self.scaling_factor), int(32/self.scaling_factor), int(64/self.scaling_factor)], [c_in, int(64/self.scaling_factor), int(64/self.scaling_factor), 
                            int(128/self.scaling_factor)]],
                use_xyz=use_xyz,
            )
        )
        c_out_1 = int(64/self.scaling_factor) + int(128/self.scaling_factor)   # 512 x c_out_1

        c_in = c_out_1
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=int(512/self.scaling_factor),
                radii=[2, 4],
                nsamples=[int(8/self.scaling_factor), int(32/self.scaling_factor)],
                mlps=[[c_in, int(32/self.scaling_factor), int(32/self.scaling_factor), int(64/self.scaling_factor)], [c_in, int(64/self.scaling_factor), int(64/self.scaling_factor), 
                    int(128/self.scaling_factor)]],
                use_xyz=use_xyz,
            )
        )
        c_out_2 = int(64/self.scaling_factor) + int(128/self.scaling_factor)   # 512 x c_out_

        c_in = c_out_2
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=int(256/self.scaling_factor),
                radii=[3, 6],
                nsamples=[int(16/self.scaling_factor), int(32/self.scaling_factor)],
                mlps=[[c_in, int(64/self.scaling_factor), int(64/self.scaling_factor), int(128/self.scaling_factor)], [c_in, int(64/self.scaling_factor), int(64/self.scaling_factor),
                 int(128/self.scaling_factor)]],
                use_xyz=use_xyz,
            )
        )
        c_out_3 = int(128/self.scaling_factor) + int(128/self.scaling_factor) # 256 x c_out_2

        # c_in = c_out_2
        # self.SA_modules.append(
        #     PointnetSAModuleMSG(
        #         npoint=16,
        #         radii=[0.4, 0.8],
        #         nsamples=[16, 32],
        #         mlps=[[c_in, 256, 256, 512], [c_in, 256, 384, 512]],
        #         use_xyz=use_xyz,
        #     )
        # )
        # c_out_3 = 512 + 512

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[int(128/self.scaling_factor) + input_channels, int(128/self.scaling_factor), int(128/self.scaling_factor), int(128/self.scaling_factor)]))
        self.FP_modules.append(PointnetFPModule(mlp=[int(256/self.scaling_factor) + c_out_1, int(128/self.scaling_factor), int(128/self.scaling_factor)])) # FP 2 from ___ FP1: (256, 256) ___ to another two-layer MLP with kernel
        self.FP_modules.append(PointnetFPModule(mlp=[int(c_out_2 + c_out_3), int(256/self.scaling_factor), int(256/self.scaling_factor)]))   # FP1 from last ___MSG: (256, c_out_2)___ to a two-layer MLP with kernel (256, 256)

        self.FC_layer = (
            pt_utils.Seq(int(128/self.scaling_factor))   ### Input channels 
            .conv1d(int(256/self.scaling_factor), bn=True)   ### 1d Conv1
            .dropout()   ### default is 0.5
            .conv1d(int(128/self.scaling_factor)) ### 1d Conv2
            .dropout()
            .conv1d(num_classes, activation=None)
        )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud):
        # type: (Pointnet2MSG, torch.cuda.FloatTensor) -> pt_utils.Seq
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]  ## This is the 0th layer in the list, the input.
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)


        for i in range(-1, -(len(self.FP_modules) + 1), -1):        #### For the baseline pointnet, TODO: check the layer index so that remove the FP1 skip connection
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return self.FC_layer(l_features[0]).transpose(1, 2).contiguous()


if __name__ == "__main__":
    from torch.autograd import Variable
    import numpy as np
    import torch.optim as optim

    B = 2
    N = 32
    inputs = torch.randn(B, N, 6).cuda()
    labels = torch.from_numpy(np.random.randint(0, 3, size=B * N)).view(B, N).cuda()
    model = Pointnet2MSG(3, input_channels=3)
    model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    print("Testing with xyz")
    model_fn = model_fn_decorator(nn.CrossEntropyLoss())
    for _ in range(5):
        optimizer.zero_grad()
        _, loss, _ = model_fn(model, (inputs, labels))
        loss.backward()
        print(loss.data[0])
        optimizer.step()

    # with use_xyz=False
    inputs = torch.randn(B, N, 6).cuda()
    labels = torch.from_numpy(np.random.randint(0, 3, size=B * N)).view(B, N).cuda()
    model = Pointnet2MSG(3, input_channels=3, use_xyz=False)
    model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    print("Testing without xyz")
    model_fn = model_fn_decorator(nn.CrossEntropyLoss())
    for _ in range(5):
        optimizer.zero_grad()
        _, loss, _ = model_fn(model, (inputs, labels))
        loss.backward()
        print(loss.data[0])
        optimizer.step()
