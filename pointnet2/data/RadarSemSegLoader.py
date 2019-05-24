from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.utils.data as data
import numpy as np
import os
import h5py
import subprocess
import shlex

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _get_data_files(list_filename):
    with open(list_filename) as f:
        return [line.rstrip() for line in f]


def _load_data_file(name):
    f = h5py.File(name)
    data = f["data"][:]
    label = f["label"][:] 
    # data = np.insert(data, 2, 0, axis=2)
    print(data.shape)
    return data, label


class RadarLowLvlSemSeg(data.Dataset):
    def __init__(self, num_points, train=True, train_rat=0.6, data_precent=1.0):
        super().__init__()
        self.data_precent = data_precent
        self.folder = "RadarLowLevel"
        self.data_dir = os.path.join(BASE_DIR, self.folder)

        self.train, self.num_points = train, num_points

        all_files = _get_data_files(os.path.join(self.data_dir, "all_files.txt"))


        data_batchlist, label_batchlist = [], []
        for f in all_files:
            data, label = _load_data_file(os.path.join(BASE_DIR, f))
            data_batchlist.append(data)
            label_batchlist.append(label)

        data_batches = np.concatenate(data_batchlist, 0)
        labels_batches = np.concatenate(label_batchlist, 0)
        num_frame = data_batches.shape[0]

        train_end_ind  = np.round(data_batches.shape[0] * train_rat).astype(np.int)
        train_idxs = np.arange(train_end_ind)
        test_idxs = np.arange(train_end_ind, num_frame)

        if self.train:
            self.points = data_batches[train_idxs, ...]
            self.labels = labels_batches[train_idxs, ...]
        else:
            self.points = data_batches[test_idxs, ...]
            self.labels = labels_batches[test_idxs, ...]

    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.num_points)
        # np.random.shuffle(pt_idxs)

        current_points = torch.from_numpy(self.points[idx, pt_idxs].copy()).type(
            torch.FloatTensor
        )
        current_labels = torch.from_numpy(self.labels[idx, pt_idxs].copy()).type(
            torch.LongTensor
        )

        return current_points, current_labels

    def __len__(self):
        return int(self.points.shape[0] * self.data_precent)

    def set_num_points(self, pts):
        self.num_points = pts

    def randomize(self):
        pass


if __name__ == "__main__":
    dset = RadarLowLvlSemSeg(16, train=True)
    print(dset[0])
    print(len(dset))
    dloader = torch.utils.data.DataLoader(dset, batch_size=32, shuffle=True)
    for i, data in enumerate(dloader, 0):
        inputs, labels = data
        if i == len(dloader) - 1:
            print(inputs.size())
