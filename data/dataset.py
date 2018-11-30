# Author: Xinshuo
# Email: xinshuow@cs.cmu.edu

import os
from torch.utils.data import Dataset
from .preprocess import *

class LipreadingDataset(Dataset):
    """BBC Lip Reading dataset."""

    def build_file_list(self, dir, set):
        labels = os.listdir(dir)            # list of words
        completeList = []
        for i, label in enumerate(labels):
            dirpath = dir + "/{}/{}".format(label, set)
            # print(i, label, dirpath)
            files = os.listdir(dirpath)
            for file in files:
                if file.endswith("mp4"):
                    filepath = dirpath + "/{}".format(file)
                    entry = (i, filepath)
                    completeList.append(entry)

        return labels, completeList

    def __init__(self, directory, set, augment=False):
        self.label_list, self.file_list = self.build_file_list(directory, set)
        self.augment = augment

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        #load video into a tensor
        label, filename = self.file_list[idx]
        vidframes = load_video(filename)
        temporalvolume = bbc(vidframes, self.augment)
        sample = {'temporalvolume': temporalvolume, 'label': torch.LongTensor([label])}
        return sample, filename