import tb
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.nn import Parameter
import pandas as pd
import datetime
import DBcheck as db
from ray import tune
import scipy.interpolate as inter
import netcdfreader as nc
import glob
import adabound
import scipy.signal as sig
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import matplotlib.pyplot as plt
import matplotlib.dates as dt
pd.options.mode.chained_assignment = None


class custdata(Dataset):
    """
     Constructor gets matrices of lat/lon coordinates ready for data.
    """
    def __init__(self,date_list,kwargs):

        self.remove_features = kwargs
        self.date_list = date_list
        self.data = [db.canakQuery(i, self.remove_features) for i in date_list]
        self.data_len = len(date_list)

    """
    __getitem__() queries database for day's data
    """
    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return self.data_len


class LSTMNet(nn.Module):
    def __init__(self, input_size,hidden_size):
        super(LSTMNet, self).__init__()
        self.LSTM = nn.LSTM(input_size,hidden_size)
    def forward(self, x):
        x1 = self.LSTM(x)
        return x1


class betterLSTM(nn.Module):
    def __init__(self, x, y, hiddensz):
        super().__init__()
        self.x = x
        self.y = y
        self.hiddensz = hiddensz

        # input gate
        self.Wx = Parameter(torch.Tensor(4, hiddensz, x, x))
        self.Wh = Parameter(torch.Tensor(4, hiddensz, x, x))
        self.Wy = nn.Conv2d(6, 1, kernel_size=3, padding=1)
        self.b = Parameter(torch.Tensor(4, hiddensz, x, y))
        # paramter initialization
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x, initstates=None):
        m = len(x)
        hidden_seq = []
        # if initstates is None:
        #     ht, ct = torch.zeros(self.hiddensz, self.x, self.y), torch.zeros(self.hiddensz, self.x, self.y)
        # else:
        ht, ct = initstates
        xt = x
        gates = self.Wx @ xt + self.Wh @ ht + self.b
        it = torch.sigmoid(gates[0])
        ft = torch.sigmoid(gates[1])
        gt = torch.tanh(gates[2])
        ot = torch.sigmoid(gates[3])
        ct = ft * ct + it * gt
        ht = ot * torch.tanh(ct)
        output = self.Wy(ht.unsqueeze(0))
        return output, (ht, ct)

class UnetLSTM(nn.Module):
    def __init__(self, n_features, n_classes, hidden,bilinear=True):
        super(UnetLSTM, self).__init__()
        self.LSTM = betterLSTM(95,477,hidden)
    def forward(self, x,hidden=None):
        lstm = self.LSTM(x,hidden)


        return lstm



num_classes = 1

decrease = 40
rate = .1
num_features = 6
hiddensz = 1
model = UnetLSTM(num_features,num_classes,hiddensz)
model.cuda()

learning_rate = .001
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=.9, nesterov=True)

start = datetime.datetime(2016, 1, 1, 6, 0, 0)
end = datetime.datetime(2017, 1, 1, 6, 0, 0)
train_dates = [start + datetime.timedelta(days=i) for i in range((end - start).days)]

print('loading training')
train = custdata(train_dates, [])
train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=1, shuffle=False, drop_last=False)
print('loading testing')
start = datetime.datetime(2018, 1, 1, 6, 0, 0)
end = datetime.datetime(2019, 1, 1, 6, 0, 0)
test_dates = [start + datetime.timedelta(days=i) for i in range((end - start).days)]
test = custdata(test_dates, [])
test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=1, shuffle=False)
print('training')

# truncate = 30
# print("allocated",torch.cuda.memory_allocated(),"max allocated",torch.cuda.max_memory_allocated(),"cached",torch.cuda.memory_cached(),"max cached",torch.cuda.max_memory_cached())
for j in range(50):
    i = 1
    # loss = 0
    hidden = torch.zeros(hiddensz, 95, 477).cuda(), torch.zeros(hiddensz, 95, 477).cuda()
    for images, labels in train_loader:
        # if i % truncate == 0:
        #     optimizer.zero_grad()
        #     loss.backward(retain_graph=True)
        #     optimizer.step()
        #     del loss
        #     torch.cuda.empty_cache()
        #     loss = 0
        #     hidden = hidden[0].data,hidden[1].data
        #     torch.cuda.empty_cache()
        images = images.cuda()
        torch.cuda.empty_cache()
        output,hidden = model(images,hidden)
        mask = (labels != -1)
        loss = criterion(output.squeeze(1).cpu()[mask], labels[mask])
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        # i+=1


        torch.cuda.empty_cache()
        
    print("Epoch:",j,"loss",loss.item())
    if loss != 0:
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        del loss
        torch.cuda.empty_cache()