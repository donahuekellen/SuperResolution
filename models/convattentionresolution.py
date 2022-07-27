import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd.variable import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
import datetime
from torch.utils.data import Dataset, DataLoader
import math

# height = 507
# width = 900

height = 450
width = 3600

class self_attention(nn.Module):
    def __init__(self, dim, enc_dim, dropout=.1,dim1 = 1,dim2=0):
        super().__init__()
        self.wq = nn.Conv2d(dim, enc_dim,1,1,0)
        self.wk = nn.Conv2d(dim, enc_dim,dim1,1,dim2)
        self.wv = nn.Conv2d(dim, enc_dim,dim1,1,dim2)
        self.dropout = nn.Dropout2d(dropout)
        self.scaler = np.sqrt(enc_dim)
        self.soft1 = nn.Softmax(-1)
        self.soft2 = nn.Softmax(-2)
        self.soft2d = nn.Softmax2d()

    def QKV(self, x):
        Q = self.wq(x)
        K = self.wk(x)
        V = self.wv(x)
        return Q, K, V

    def score(self, Q, K, V, mask):
        scores = Q@K.transpose(-2,-1) / self.scaler
        temp1 = self.dropout(self.soft1(scores))
        temp2 = self.dropout(self.soft2(Q.transpose(-2,-1)@K/self.scaler))

        return temp1@V@temp2

    def forward(self, x, mask=None):
        Q, K, V = self.QKV(x)
        return self.score(Q, K, V, mask)

class encdec_attention(nn.Module):
    def __init__(self, dim, dropout=.1):
        super().__init__()
        self.wq = nn.Conv2d(dim, dim,1,1,0)
        self.wk = nn.Conv2d(dim, dim,3,1,1)
        self.wv = nn.Conv2d(dim, dim,3,1,1)
        self.dropout = nn.Dropout2d(dropout)
        self.scaler = np.sqrt(dim)
        self.soft1 = nn.Softmax(-1)
        self.soft2 = nn.Softmax(-2)
        self.soft2d = nn.Softmax2d()

    def QKV(self, x,y):
        Q = self.wq(x)
        K = self.wk(y)
        V = self.wv(y)
        return Q, K, V

    def score(self, Q, K, V, mask):  
        scores = Q@K.transpose(-2,-1) / self.scaler
        temp1 = self.dropout(self.soft1(scores))
        temp2 = self.dropout(self.soft2(Q.transpose(-2,-1)@K/self.scaler))
        return temp1@V@temp2

    def forward(self, x,y, mask=None):
        Q, K, V = self.QKV(x,y)
        return self.score(Q, K, V, mask)



class encoder(nn.Module):
    def __init__(self, dim, enc_dim, dropout=.1):
        super().__init__()

        self.attention = self_attention(dim, enc_dim, dropout,3,1)
        self.norm1 = nn.LayerNorm([enc_dim,height,width])
        # self.norm1 = nn.BatchNorm2d(enc_dim)
        # self.norm1 = nn.InstanceNorm2d(enc_dim)
        self.linear = nn.Sequential(
            nn.Conv2d(enc_dim, enc_dim,3,1,1),
            nn.ReLU()
        )
        self.drop = nn.Dropout2d(dropout)
        self.residual = nn.Conv2d(dim, enc_dim,3,1,1)
        self.norm2 = nn.LayerNorm([enc_dim, height,width])
        # self.norm2 = nn.BatchNorm2d(enc_dim)
        # self.norm2 = nn.InstanceNorm2d(enc_dim)

    def forward(self, x):
        z = self.attention(x)
        z = self.norm1(z + self.residual(x))
        # z = self.residual(x) + z
        z2 = self.drop(self.linear(z))
        return self.norm2(z + z2)


class decoder(nn.Module):
    def __init__(self, dim, enc_dim, dropout=.1):
        super().__init__()

        self.attention = self_attention(dim, enc_dim, dropout,3,1)
        self.norm1 = nn.LayerNorm([enc_dim,height,width])
        # self.norm1 = nn.BatchNorm2d(enc_dim)
        # self.norm1 = nn.InstanceNorm2d(enc_dim)
        self.linear = nn.Sequential(
            nn.Conv2d(enc_dim, enc_dim,3,1,1),
            nn.ReLU()
        )
        self.EDattention = encdec_attention(enc_dim, dropout)
        self.norm2 = nn.LayerNorm([enc_dim,height,width])
        # self.norm2 = nn.BatchNorm2d(enc_dim)
        # self.norm2 = nn.InstanceNorm2d(enc_dim)
        self.residual = nn.Conv2d(dim, enc_dim,3,1,1)
        self.drop = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm([enc_dim,height,width])
        # self.norm3 = nn.BatchNorm2d(enc_dim)
        # self.norm3 = nn.InstanceNorm2d(enc_dim)

    def forward(self, x,y):
        z = self.attention(x)
        z = self.norm1(z + self.residual(x))
        # z = z + self.residual(x)
        z2 = self.EDattention(z, y)
        z2 = self.norm2(z2 + z)
        # z2 = z2 + z
        z3 = self.drop(self.linear(z2))
        return self.norm3(z3 + z2)



class transformer(nn.Module):
    def __init__(self, dim, dec_dim, enc_dim):
        super().__init__()

        self.encoders = []
        DROP = .1
        self.encoders.append(encoder(dim, enc_dim,DROP))
        self.encoders.append(encoder(enc_dim, enc_dim, DROP))
        self.encoders.append(encoder(enc_dim, enc_dim, DROP))
        self.encoders.append(encoder(enc_dim, enc_dim, DROP))
        # self.encoders.append(encoder(enc_dim, enc_dim, DROP))
        # self.encoders.append(encoder(enc_dim, enc_dim, DROP))
        # self.encoders.append(encoder(enc_dim, enc_dim, DROP))


        self.encoders = nn.ModuleList(self.encoders)

        self.decoders = []
        # self.decoders.append(decoder(dec_dim, enc_dim,DROP))
        # self.decoders.append(decoder(enc_dim, enc_dim, DROP))
        # self.decoders.append(decoder(enc_dim, enc_dim, DROP))
        # self.decoders.append(decoder(enc_dim, enc_dim, DROP))
        # self.decoders.append(decoder(enc_dim, enc_dim, DROP))
        self.decoders = nn.ModuleList(self.decoders)

        self.norm1 = nn.LayerNorm([enc_dim,height,width])
        self.final = nn.Sequential(
            nn.Conv2d(enc_dim, 1,1)
        )



    def encode(self, x):
        for layer in self.encoders:
            x = layer(x)
        return x

    def decode(self, y,x):
        for layer in self.decoders:
            y = layer(y,x)
        return y

    def forward(self, x,y):
        x = self.encode(x)
        return self.final(x)