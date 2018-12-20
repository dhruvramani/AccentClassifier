import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self, num_classes=5):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 3 * 3, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        # print(x.shape)
        x = x.view(x.size()[0], 256 * 3 * 3)
        x = self.classifier(x)
        return x

class CNNNet(nn.Module):
    def __init__(self, num_classes=15):
        super(CNNNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # nn.Conv2d(192, 384, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        # print(x.shape)
        x = x.view(x.size()[0], 256 * 7 * 7)
        x = self.classifier(x)
        return x

class PrathamNetwork(nn.Module):
    def __init__(self):
        super(PrathamNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(20*29*29, 4096)
        self.fc2 = nn.Linear(4096, 5)

        # self.enc.load_state_dict(torch.load('../save/checkpoint/.ckpt'))


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # print(x.shape)
        x = x.view(x.size()[0], 20 * 29 * 29)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class exp1(nn.Module):
    def __init__(self):
        super(exp1,self).__init__()
        #Mel Spectoram(batch,1,128,500) ->  
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.25)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 5)
        )

    def forward(self,x):
        x = self.features(x)
        # print(x.shape)
        x = x.view(x.size()[0], 64 * 7 * 7)
        x = self.classifier(x)
        return x

class exp2(nn.Module):
    def __init__(self):
        super(exp2,self).__init__()
        #Mel Spectoram(batch,1,128,500) ->  
        self.features = nn.Sequential(
            nn.Conv1d(1, 20, 20, padding=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=1, padding=0),
            nn.Conv1d(20, 50, 50, padding=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4, stride=1, padding=0),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 1 * 1, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 15)
        )

    def forward(self,x):
        x = self.features(x)
        print(x.shape)
        x = x.view(x.size()[0], 64 * 1 * 1)
        x = self.classifier(x)
        return x

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class exp3(torch.nn.Module):
    def __init__(self, n_channels=1):
        super(exp3, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.e1 = down(64, 128)
        self.e2 = down(128, 256)
        self.e3 = down(256, 256)
        self.c1 = nn.Linear(256*16*16, 4096)
        self.c2 = nn.Linear(4096, 5)

    def forward(self, x):
        x = self.inc(x)
        x = self.e1(x)
        x = self.e2(x)
        x = self.e3(x)
        print(x.shape)
        x = x.view(x.size()[0], 256 * 16 * 16)
        x = F.relu(self.c1(x))
        x = self.c2(F.dropout(x, 0.25))
        return x

