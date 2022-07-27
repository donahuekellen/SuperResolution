import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from unet import Up,DoubleConv

class convLSTM(nn.Module):
    def __init__(self, hiddensz,inchannels,outchannels,kernel=3,padding=1):
        super().__init__()

        self.hiddensz = hiddensz
        self.ht,self.ct = None,None

        self.Wxiconv = nn.Conv2d(inchannels, hiddensz, kernel, padding=padding)
        self.Whiconv = nn.Conv2d(hiddensz, hiddensz, kernel, padding=padding)
        self.Wxfconv = nn.Conv2d(inchannels, hiddensz, kernel, padding=padding)
        self.Whfconv = nn.Conv2d(hiddensz, hiddensz, kernel, padding=padding)
        self.Wxgconv = nn.Conv2d(inchannels, hiddensz, kernel, padding=padding)
        self.Whgconv = nn.Conv2d(hiddensz, hiddensz, kernel, padding=padding)
        self.Wxoconv = nn.Conv2d(inchannels, hiddensz, kernel, padding=padding)
        self.Whoconv = nn.Conv2d(hiddensz, hiddensz, kernel, padding=padding)
        self.Wy = nn.Conv2d(hiddensz, outchannels, kernel_size=kernel, padding=padding)
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)
    def clear_grad(self):
        self.ht,self.ct = self.ht.data,self.ct.data

    def reset_hidden(self):
        self.ht,self.ct = None,None

    def forward(self, x):
        if self.ht is None:
            self.ht,self.ct = torch.zeros(x.shape[0], self.hiddensz, x.shape[-2], x.shape[-1]).cuda(), torch.zeros(x.shape[0], self.hiddensz, x.shape[-2], x.shape[-1]).cuda()
        it = torch.sigmoid(self.Wxiconv(x)+self.Whiconv(self.ht))
        ft = torch.sigmoid(self.Wxfconv(x)+self.Whfconv(self.ht))
        gt = torch.tanh(self.Wxgconv(x)+self.Whgconv(self.ht))
        ot = torch.sigmoid(self.Wxoconv(x)+self.Whoconv(self.ht))
        self.ct = ft * self.ct + it * gt
        self.ht = ot * torch.tanh(self.ct)
        output = self.Wy(self.ht)
        return output


class lstmconv(nn.Module):
    def __init__(self, inch, outch,kernel=3,padding=1):
        super().__init__()
        hiddensz = int(np.ceil(np.sqrt(inch*outch)))
        self.d = nn.Sequential(
            convLSTM(hiddensz, inch, outch,kernel,padding),
            # nn.ELU(),
            nn.BatchNorm2d(outch),
            nn.Conv2d(outch, outch, kernel, 1, padding),
            nn.ELU(),
            nn.BatchNorm2d(outch),
        )

    def clear_grad(self):
        self.d[0].clear_grad()
    def reset_hidden(self):
        self.d[0].reset_hidden()
    def forward(self, x):
        return self.d(x)

class downlstm(nn.Module):
    down_count = 0
    def __init__(self,inch,outch):
        super().__init__()
        # self.down_count+=1
        # x,y = outsizeCalc(self.down_count)
        hiddensz = int(round(np.sqrt(inch * outch)))
        self.poolconv = nn.Sequential(
            nn.MaxPool2d(2),
            convLSTM(hiddensz, inch, outch),
            nn.BatchNorm2d(outch),
        )

    def clear_grad(self):
        self.poolconv[1].clear_grad()

    def reset_hidden(self):
        self.poolconv[1].reset_hidden()
    def forward(self,x):
        return self.poolconv(x)

class uplstm(nn.Module):
    def __init__(self,inch,outch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.lstm = convLSTM(int(round(np.sqrt(inch * outch))), inch, outch)

    def clear_grad(self):
        self.lstm.clear_grad()

    def reset_hidden(self):
        self.lstm.reset_hidden()
    def forward(self,x1,x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.lstm(x)

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.layers = []


        # self.layers.append(lstmconv(6, 12))
        # self.layers.append(downlstm(12,24))
        # self.layers.append(downlstm(24, 48))
        # self.layers.append(lstmconv(48, 48))
        # self.layers.append(uplstm(96, 24))
        # self.layers.append(uplstm(48, 12))
        # self.layers.append(uplstm(24, 6))
        # self.layers.append(lstmconv(6, 1))
        # self.layers.append(lstmconv(1,1,1,0))
        # self.layers.append(lstmconv(7, 10))
        # self.layers.append(lstmconv(10, 13))
        # self.layers.append(lstmconv(13, 16))
        # self.layers.append(lstmconv(16, 19))
        # self.layers.append(lstmconv(19, 22))
        # self.layers.append(lstmconv(22, 25))
        # self.layers.append(lstmconv(25, 28))
        # self.layers.append(lstmconv(28, 31))
        # self.layers.append(lstmconv(31, 22))
        # self.layers.append(lstmconv(22, 13))
        # self.layers.append(lstmconv(13, 7))
        # self.layers.append(lstmconv(7, 1))
        # self.layers.append(lstmconv(1, 1,1,0))

        # self.layers.append(lstmconv(6,32))
        # self.layers.append(downlstm(32,48))
        # self.layers.append(lstmconv(48, 48))
        # self.layers.append(Up(96,32))
        # self.layers.append(Up(64,32))
        # self.layers.append(convLSTM(32, 32,1,1,0))

        # self.layers.append(lstmconv(6, 6))
        self.layers.append(lstmconv(6,8))
        self.layers.append(lstmconv(8, 10))
        self.layers.append(lstmconv(10, 12))
        self.layers.append(convLSTM(12, 12,1,1,0))
        # self.layers.append(lstmconv(1,1,1,0))

        self.layers = nn.ModuleList(self.layers)


    def clear_grad(self):
        for i in self.layers:
            i.clear_grad()
    def reset_hidden(self):
        for i in self.layers:
            i.reset_hidden()
    def forward(self, x):
        x = self.layers[0](x)
        x = self.layers[1](x)
        x = self.layers[2](x)
        x = self.layers[3](x)
       

        return x