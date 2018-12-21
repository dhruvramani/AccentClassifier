import gc
import os
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from sklearn.model_selection import train_test_split
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from utils import *

DEBUG = True
SILENCE_THRESHOLD = .01
RATE = 24000
N_MFCC = 13
COL_SIZE = 30
EPOCHS = 10#35#250

class AccentDataset(Dataset):
    """Accent dataset."""

    def __init__(self, X_train, y_train):
        """
        Args:
            csv_file (string): Path to the csv file.
            root_dir (string): Directory with all the recordings.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.count = 0

    def __len__(self):
        return Counter(y_train)

    def __getitem__(self, idx):
        sample = (torch.Tensor(X_train[self.count]), torch.Tensor(y_train[self.count])
        self.count += 1
        return sample

class AlexNet(nn.Module):
    def __init__(self, num_classes=3):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
            nn.BatchNorm2d(128),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 3 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        print(x.shape)
        x = x.view(x.size()[0], 256 * 3 * 3)
        x = self.classifier(x)
        return x

 def make_segments(mfccs,labels):
    '''
    Makes segments of mfccs and attaches them to the labels
    :param mfccs: list of mfccs
    :param labels: list of labels
    :return (tuple): Segments with labels
    '''
    segments = []
    seg_labels = []
    for mfcc,label in zip(mfccs,labels):
        for start in range(0, int(mfcc.shape[1] / COL_SIZE)):
            segments.append(mfcc[:, start * COL_SIZE:(start + 1) * COL_SIZE])
            seg_labels.append(label)
    return(segments, seg_labels)


def segment_one(mfcc):
    '''
    Creates segments from on mfcc image. If last segments is not long enough to be length of columns divided by COL_SIZE
    :param mfcc (numpy array): MFCC array
    :return (numpy array): Segmented MFCC array
    '''
    segments = []
    for start in range(0, int(mfcc.shape[1] / COL_SIZE)):
        segments.append(mfcc[:, start * COL_SIZE:(start + 1) * COL_SIZE])
    return(np.array(segments))

def create_segmented_mfccs(X_train):
    '''
    Creates segmented MFCCs from X_train
    :param X_train: list of MFCCs
    :return: segmented mfccs
    '''
    segmented_mfccs = []
    for mfcc in X_train:
        segmented_mfccs.append(segment_one(mfcc))
    return(segmented_mfccs)

net = AlexNet()
criterion = nn.CrossEntropyLoss(reduction='sum') # To calculate the average later

def train(epoch, X_train, y_train):
    
    trainset = AccentDataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    dataloader = iter(dataloader)
    print('\nEpoch: %d' % epoch)
    
    train_loss, correct, total = 0, 0, 0
    params = net.parameters()
    optimizer = optim.Adam(params, lr=0.001)#, momentum=0.9)#, weight_decay=5e-4)

    for batch_idx in range(start_step, len(dataloader)):
        (inputs, targets) = next(dataloader)
        inputs, targets = inputs[0], targets[0] # batch_size == 1 ~= 1 sample
        targets = targets.type(torch.LongTensor)
        inputs, targets = inputs.to(device), targets.to(device)

        # NOTE : Main optimizing here
        optimizer.zero_grad()
        y_pred = net(inputs)
        loss = criterion(y_pred, targets)
        loss = loss / inputs.shape[0]
        loss.backward()
        optimizer.step()

        # NOTE : Logging here
        train_loss += loss.item()
        _, predicted = y_pred.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        #print(accuracy(y_pred, targets))

        gc.collect()
        torch.cuda.empty_cache()

        progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

if __name__ == '__main__':
    '''
        Console command example:
        python trainmodel.py bio_metadata.csv model50
        '''

    # Load arguments
    file_name = sys.argv[1]
    model_filename = sys.argv[2]

    # Load metadata
    df = pd.read_csv(file_name)

    # Filter metadata to retrieve only files desired
    filtered_df = getsplit.filter_df(df)

    # Get resampled wav files using multiprocessing
    if DEBUG:
        print('loading wav files')

    # Train test split
    X_train, X_test, y_train, y_test = getsplit.split_people(filtered_df)
    
    # Get statistics
    train_count = Counter(y_train)
    test_count =  Counter(y_test)
    
    # Create segments 
    if DEBUG:
        print('converting to segments')
    X_train, y_train = make_segments(X_train, y_train)

    # Randomize training segments
    X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=0)
    
    #Trying some shit
    for epoch in range(1,5):
    	train(epoch, X_train, y_train)
    # model = train_model(np.array(X_train), np.array(y_train), np.array(X_test),np.array(y_test))

    # Make predictions on full X_test MFCCs
    y_predicted = accuracy.predict_class_all(create_segmented_mfccs(X_test), model)

    # Print statistics
    print(train_count)
    print(test_count)
    # print( acc_to_beat)
    print(np.sum(accuracy.confusion_matrix(y_predicted, y_test),axis=1))
    print(accuracy.confusion_matrix(y_predicted, y_test))
    print(accuracy.get_accuracy(y_predicted,y_test))

