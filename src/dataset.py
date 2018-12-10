import os
import torch
import librosa
import numpy as np
import pandas as pd
from feature import *
from torch.utils.data import Dataset, DataLoader
import torchaudio
import librosa
import matplotlib
import matplotlib.pyplot as plt



#    vdataset = VCTK('/home/nevronas/dataset/', download=False, transform=inp_transform)
#    dataloader = DataLoader(vdataset, batch_size=args.batch_size, shuffle=True,  collate_fn=collate_fn)
def read_audio(fp, downsample=True):
    sig, sr = torchaudio.load(fp)
    if downsample:
        # 48khz -> 16 khz
        if sig.size(0) % 3 == 0:
            sig = sig[::3].contiguous()
        else:
            sig = sig[:-(sig.size(0) % 3):3].contiguous()
    return sig, sr

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
        inp = read_audio(inp)[0]
        inp = inp.numpy()
        inp = inp.flatten()
        inp = transform_stft_new(inp)
        # matplotlib.image.imsave('../save/imgs/stft_new.png', inp)
        print(inp.shape)
        foo = np.expand_dims(inp, axis=2)
        aud = np.concatenate((foo, np.zeros(foo.shape), np.zeros(foo.shape)), axis=2)
        print(aud.shape)
        plt.imshow(aud)
        plt.show()
        _ = input()
                
        # lbl = np.zeros((15))
        # lbl[label] = 1
        for j in range(0,inp.shape[1],500):
            try:
                sam = inp[:,j:j+500]
                if(sam.shape[1]<500):
                    sam = librosa.util.pad_center(sam, 500)
                # print(sam.shape)
                audio = inp
                # plt.imshow(audio)
                # plt.show()
                # _ = input()
                
                aud_sample.append(sam)  
                class_sample.append(label)
            except Exception as e:
                print(str(e))
                pass
    aud_sample = torch.Tensor(aud_sample)
    class_sample = torch.Tensor(class_sample)
    aud_sample = aud_sample.unsqueeze(1)

    return aud_sample, class_sample


class AccentDataset(Dataset):
    """Accent dataset."""

    def __init__(self, csv_file="/home/nevronas/dataset/accent/speakers_all.csv", root_dir="/home/nevronas/dataset/accent/recordings", batch_size=10, transform=inp_transform):
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
            if(str(row['native_language'].values[0]) in self.top_15_langs):
                filename = row['filename'].values[0]
                filename = "{}/{}.mp3".format(self.root_dir, filename)
                audios.append({"audio" : filename, "class" : self.top_15_langs.index(row['native_language'].values[0])})
                count += 1
            i += 1

        return audios

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, idx):
        self.count += 1
        sample = self.get_data()
        if self.transform:
            sample = self.transform(sample)

        return sample
