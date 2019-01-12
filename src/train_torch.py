import gc
import os
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from collections import Counter
from sklearn.model_selection import train_test_split
import pickle

from segmentation import *
from utils import *
from dataset import *
from model import *
from feature import * 
import accuracy

parser = argparse.ArgumentParser(description='PyTorch Accent Classifier')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--epochs', default=10, type=int, help='number of epochs to run')
parser.add_argument('--resume', '-r', default=0, type=int, help='resume from checkpoint')
parser.add_argument('--preparedata', type=bool, default=False, help='Recreate the dataset.')
args = parser.parse_args()

FILE_NAME = 'data.csv'

best_acc, start_epoch, start_step = 0, 0, 0  # best test accuracy, start from epoch 0 or last checkpoint epoch

if(args.preparedata):
    print('==> Preparing data..')
    filtered_df = filter_df(None)
    X_train, X_test, y_train, y_test = split_people(filtered_df)

    train_count = Counter(y_train)
    test_count =  Counter(y_test)
    print('==> Creatting segments..')
    X_train, y_train = make_segments(X_train, y_train)
    X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=0)

    print("==> Saving dataset..")
    with open("../save/dataset/data.dat", "wb") as f:
        data = (X_train, X_test, y_train, y_test)
        pickle.dump(data, f)
else:
    print("==> Loading dataset..")
    with open("../save/dataset/data.dat", "rb") as f:
        (X_train, X_test, y_train, y_test) = pickle.load(f)
        print(len(X_train))
        
if args.resume:
    if(os.path.isfile('../save/network.ckpt')):
        net.load_state_dict(torch.load('../save/network.ckpt'))
        print("=> Network : loaded")
    
    if(os.path.isfile("../save/info.txt")):
        with open("../save/info.txt", "r") as f:
            start_epoch, start_step = (int(i) for i in str(f.read()).split(" "))
            print("=> Network : prev epoch found")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('==> Building network..')
net = RowCNN()
criterion = nn.CrossEntropyLoss()
net = net.to(device)

def train(epoch, X_train, y_train):

    trainset = AccentDataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    dataloader = iter(dataloader)
    print('\nEpoch: %d' % epoch)
    
    train_loss, correct, total = 0, 0, 0
    params = net.parameters()
    optimizer = optim.Adam(params, lr = args.lr)#, momentum=0.9)#, weight_decay=5e-4)

    for batch_idx in range(len(dataloader)):
        inputs, targets = next(dataloader)
        inputs, targets = inputs.to(device), targets.to(device)
        # print(inputs.shape)
        # NOTE : Main optimizing here
        optimizer.zero_grad()
        y_pred = net(inputs)
        loss = criterion(y_pred, targets)
        loss.backward()
        optimizer.step()

        # NOTE : Logging here
        train_loss += loss.item()
        _, predicted = y_pred.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        with open("../save/logs/train_loss.log", "a+") as lfile:
            lfile.write("{}\n".format(train_loss / total))

        with open("../save/logs/train_acc", "a+") as afile:
            afile.write("{}\n".format(correct / total))

        gc.collect()
        torch.cuda.empty_cache()

        torch.save(net.state_dict(), '../save/network.ckpt')

        with open("../save/info.txt", "w+") as f:
            f.write("{} {}".format(epoch, batch_idx))

        progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        start_step = 0


# print('==> Preparing data..')
# # Load metadata
# df = pd.read_csv(FILE_NAME)

# # Filter metadata to retrieve only files desired
# filtered_df = filter_df(df)
# # Train test split
# X_train, X_test, y_train, y_test = split_people(filtered_df)
    
# # Get statistics
# train_count = Counter(y_train)
# test_count =  Counter(y_test)


# print('==> Creatting segments..')
# # Create segments
# X_train, y_train = make_segments(X_train, y_train)

# # Randomize training segments
# X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=0)
    
# print(X_train.shape)
#Training
for epoch in range(start_epoch, start_epoch + args.epochs):
    train(epoch, X_train, y_train)


print('==> Testing network..')
# Make predictions on full X_test mels
y_predicted = accuracy.predict_class_all(create_segmented_mels(X_test), net)

# Print statistics
# print(train_count)
# print(test_count)
print(np.sum(accuracy.confusion_matrix(y_predicted, y_test),axis=1))
print(accuracy.confusion_matrix(y_predicted, y_test))
print(accuracy.get_accuracy(y_predicted,y_test))

