import os
import torch
import librosa
import numpy as np
import pandas as pd
from feature import *
from torch.utils.data import Dataset, DataLoader

#    vdataset = VCTK('/home/nevronas/dataset/', download=False, transform=inp_transform)
#    dataloader = DataLoader(vdataset, batch_size=args.batch_size, shuffle=True,  collate_fn=collate_fn)

def collate_fn(data):
    data = list(filter(lambda x: type(x[1]) != int, data))
    audios, captions = zip(*data)
    data = None
    del data
    audios = torch.stack(audios, 0)
    return audios, captions

def inp_transform(sample):
    aud_sample, class_sample = [], []
    for i in sample:
        inp, label = i['audio'], i['class']
        inp = inp.flatten()
        inp = transform_stft(inp)
        #inp = torch.Tensor(inp)
        #inp = inp.unsqueeze(0)
        lbl = np.zeros((15))
        lbl[label] = 1
        aud_sample.append(inp)  
        class_sample.append(lbl)

    aud_sample = torch.Tensor(aud_sample)
    class_sample = torch.Tensor(class_sample)
    return aud_sample, class_sample


class AccentDataset(Dataset):
    """Accent dataset."""

    def __init__(self, csv_file="/home/nevronas/dataset/accent/speaker_all.csv", root_dir="/home/nevronas/dataset/accent/recordings", batch_size=10, transform=inp_transform):
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
        i, count = self.count * self.batch_size, 0
        audios = []
        while(count < self.batch_size):
            row = self.csv.iloc[[int(i % self.csv.shape[0])]]
            if(row['native_language'][1] in self.top_15_langs):
                filename = row['filename'][1]
                signal, fs = librosa.load("{}/{}.mp3".format(self.root_dir, filename))
                audios.append({"audio" : signal, "class" : self.top_15_langs.index(row['native_language'][1])})
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