import os
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from model import *
from dataset import *
from utils import progress_bar
import gc
# NOTE : All parser related stuff here

parser = argparse.ArgumentParser(description='PyTorch Accent Classifier')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--resume', '-r', default=0, type=int, help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc, start_epoch, start_step = 0, 0, 0  # best test accuracy, start from epoch 0 or last checkpoint epoch

# NOTE : All data related stuff here

print('==> Preparing data..')
'''
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
'''

classes = ('english', 'spanish', 'arabic', 'mandarin', 'french', 'german', 'korean', 'russian', 'portuguese', 'dutch', 'turkish', 'italian', 'polish', 'japanese', 'vietnamese')

# NOTE : Build model here & check if to be resumed

print('==> Building network..')
net = AlexNet()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    if(os.path.isfile('../save/network.ckpt')):
        net.load_state_dict(torch.load('../save/network.ckpt'))
        print("=> Network : loaded")
    
    if(os.path.isfile("../save/info.txt")):
        with open("../save/info.txt", "r") as f:
            start_epoch, start_step = (int(i) for i in str(f.read()).split(" "))
            print("=> Network : prev epoch found")


# NOTE : Define losses here

criterion = nn.CrossEntropyLoss()

def train(epoch):
    global start_step
    trainset = AccentDataset()
    dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    dataloader = iter(dataloader)
    print('\nEpoch: %d' % epoch)
    
    train_loss, correct, total = 0, 0, 0
    params = net.parameters()
    optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)

    for batch_idx in range(start_step, len(dataloader)):
        (inputs, targets) = next(dataloader)
        inputs, targets = inputs[0], targets[0]
        targets = targets.type(torch.LongTensor)
        inputs, targets = inputs.to(device), targets.to(device)

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

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            _, outputs = net(inputs)
            loss = loss(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            with open("../save/logs/test_loss.log", "a+") as lfile:
                lfile.write(str(test_loss / total))
                lfile.write("\n")

            with open("../save/logs/test_acc.log", "a+") as afile:
                afile.write(str(correct / total))
                afile.write("\n")

            progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {'net': net.state_dict(), 'acc': acc, 'epoch': epoch}
        if not os.path.isdir('../save/checkpoint'):
            os.mkdir('../save/checkpoint')
        torch.save(state, '../save/checkpoint/ckpt.t7')
        best_acc = acc

for epoch in range(start_epoch, start_epoch + 200):
    train(epoch)
    test(epoch)
