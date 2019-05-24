
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

from torch.utils.data import DataLoader
import etw_pytorch_utils as pt_utils
from pointnet2.data import RadarLowLvlSemSeg

import tqdm

from copy import deepcopy





class Tester(object):
    def __init__(self, model, loss_func, use_gpu):
        self.model = model
        self.loss_func = loss_func
        self.use_gpu = use_gpu
        if use_gpu:
            self.model.cuda()
        self.model.eval()

    def test(self, dataloader):
        total_loss = 0
        cnter = 1
        pred_labels_all = np.array([])
        true_labels_all = np.array([])
        with tqdm.tqdm(total = len(dataloader), leave=False, desc='test') as pbar:
            for i, batch in enumerate(dataloader):
                pbar.update()
                pred_labels, labels, res_dict = self._test_it(batch)
                total_loss += res_dict['loss']
                cnter+=1
                pred_labels_np, labels_np = pred_labels.cpu().numpy(), labels.cpu().numpy()
                # print(pred_labels_np.shape)
                if i == 0:
                    pred_labels_all = pred_labels_np
                    true_labels_all = labels_np
                else:
                    pred_labels_all = np.concatenate((pred_labels_all, pred_labels_np), axis=0)
                    true_labels_all = np.concatenate((true_labels_all, labels_np), axis=0)
            loss = total_loss/cnter

        return loss, pred_labels_all, true_labels_all


    def _test_it(self, batch):
        self.model.zero_grad()
        classes, true_labels, res_dict = self._forward_pass(batch)
        return classes, true_labels, res_dict 

    def _forward_pass(self, batch):
        inputs, true_labels = batch
        if self.use_gpu:
            inputs = inputs.cuda()
            true_labels = true_labels.cuda()
        self.model.eval()
        preds = self.model(inputs)
        loss = self.loss_func(preds.view(true_labels.numel(),-1), true_labels.view(-1))
        _, pred_labels = torch.max(preds, -1)
        acc = (pred_labels.view(-1) == true_labels.view(-1)).float().sum() / true_labels.numel()
        res_dict = {
            "loss": loss.item(),
            "acc": acc.item()
        }
        return pred_labels, true_labels, res_dict
