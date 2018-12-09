import os
import torch
import argparse
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from model import * # NOTE : Import all the models here
from utils import progress_bar

# NOTE : All parser related stuff here

parser = argparse.ArgumentParser(description='PyTorch Audio Style Transfer')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc, start_epoch = 0, 0  # best test accuracy, start from epoch 0 or last checkpoint epoch

# NOTE : All data related stuff here

print('==> Preparing data..')
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


trainset = torchvision.datasets.CIFAR10(root='../dataset', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='../dataset', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# NOTE : Build model here & check if to be resumed

print('==> Building network..')
t_net = TransformationNetwork()
t_net = t_net.to(device)
if device == 'cuda':
    t_net = torch.nn.DataParallel(t_net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('../save/checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('../save/checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# NOTE : Define losses here

criterion = nn.CrossEntropyLoss()

def train(epoch, curr_class, old_classes):
    print('\nEpoch: %d' % epoch)
    net.train()
    
    train_loss, correct, total = 0, 0, 0
    params = net.parameters()
    optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # NOTE : Main optimizing here
        optimizer.zero_grad()
        y_pred = net(inputs)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()

        # NOTE : Logging here
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        with open("../save/logs/train_loss.log", "a+") as lfile:
            lfile.write("{}\n".format(train_loss / total))

        with open("../save/logs/train_acc", "a+") as afile:
            afile.write("{}\n".format(correct / total))

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch, curr_class):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            _, outputs = t_net(inputs, old_class=False)
            loss = loss(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            with open("./logs/test_loss_{}.log".format(curr_class), "a+") as lfile:
                lfile.write(str(test_loss / total))
                lfile.write("\n")

            with open("./logs/test_acc_{}.log".format(curr_class), "a+") as afile:
                afile.write(str(correct / total))
                afile.write("\n")

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {'net': net.state_dict(), 'acc': acc, 'epoch': epoch}
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc

# NOTE : Final running here

for epoch in range(start_epoch, start_epoch + 200):
    train(epoch, i, old_classes_arr)
    test(epoch, i)
