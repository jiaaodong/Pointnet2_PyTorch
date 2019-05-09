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

        return ModelReturn(preds, loss, {"acc": acc.item(), "loss": loss.item()})

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

        self.SA_modules = nn.ModuleList()
        c_in = input_channels
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=512, # The number of groups
                radii=[1, 3],
                nsamples=[8, 32], # The number of samples in each group 
                mlps=[[c_in, 32, 32, 64], [c_in, 64, 64, 128]],
                use_xyz=use_xyz,
            )
        )
        c_out_1 = 64 + 128   # 512 x c_out_1

        c_in = c_out_1
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=512,
                radii=[2, 4],
                nsamples=[8, 32],
                mlps=[[c_in, 32, 32, 64], [c_in, 64, 64, 128]],
                use_xyz=use_xyz,
            )
        )
        c_out_2 = 64 + 128   # 512 x c_out_

        c_in = c_out_2
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=256,
                radii=[3, 6],
                nsamples=[16, 32],
                mlps=[[c_in, 64, 64, 128], [c_in, 64, 64, 128]],
                use_xyz=use_xyz,
            )
        )
        c_out_3 = 128 + 128 # 256 x c_out_2

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
        self.FP_modules.append(PointnetFPModule(mlp=[128 + input_channels, 128, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + c_out_1, 128, 128])) # FP 2 from ___ FP1: (256, 256) ___ to another two-layer MLP with kernel
        self.FP_modules.append(PointnetFPModule(mlp=[c_out_2 + c_out_3, 256, 256]))   # FP1 from last ___MSG: (256, c_out_2)___ to a two-layer MLP with kernel (256, 256)

        self.FC_layer = (
            pt_utils.Seq(128)   ### Input channels 
            .conv1d(256, bn=True)   ### 1d Conv1
            .dropout()   ### default is 0.5
            .conv1d(128) ### 1d Conv2
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
