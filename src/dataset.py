import os
import torch
import librosa
import numpy as np
import pandas as pd
from features import *
from torch.utils.data import Dataset, DataLoader

class AccentDataset(Dataset):
    """Accent dataset."""

    def __init__(self, csv_file="/home/nevronas/dataset/accent/speaker_all.csv", root_dir="/home/nevronas/dataset/accent/recordings", batch_size=10, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file.
            root_dir (string): Directory with all the recordings.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.batch_size = batch_size
        self.top_15_langs = ['english', 'spanish', 'arabic', 'mandarin', 'french', 'german', 'korean', 'russian', 'portuguese', 'dutch', 'turkish', 'italian', 'polish', 'japanese', 'vietnamese']
        self.count = 0

    def get_data(self):
        i = self.count * self.batch_size, count = 0
        audios = []
        while(count < self.batch_size):
            row = self.csv.iloc[[int(i % self.csv.shape[0])]]
            if(row['native_language'][1] in self.top_15_langs):
                filename = row['filename'][1]
                signal, fs = librosa.load("{}/{}.mp3".format(self.root_dir, filename))
                audios.append({"audio" : signal, "class" : row['native_language'][1]})
                count += 1
            i += 1

        return audios

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self):
        self.count += 1
        sample = get_data()
        if self.transform:
            sample = self.transform(sample)

        return sample