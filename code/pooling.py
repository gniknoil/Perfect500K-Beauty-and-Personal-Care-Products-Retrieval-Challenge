import numpy as np
import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import math

__all__=['Rmac_Pooling','Mac_Pooling','SPoC_pooling','Ramac_Pooling','Grmac_Pooling']

class Rmac_Pooling(nn.Module):
    def __init__(self):
        super(Rmac_Pooling,self).__init__()

    def get_regions(self,L=[1,2,4]):
        ovr = 0.4 # desired overlap of neighboring regions
        steps = np.array([1,2, 3, 4, 5, 6], dtype=np.float) # possible regions for the long dimension

        w = min(self.W,self.H)

        b = (max(self.H,self.W) - w)/steps
        idx = np.argmin(abs(((w ** 2 - w*b)/w ** 2)-ovr)) # steps(idx) regions for long dimension

        # region overplus per dimension
        Wd, Hd = 0, 0
        if self.H < self.W:
            Wd = idx + 1
        elif self.H > self.W:
            Hd = idx + 1

        regions = []

        for l in L:

            wl = np.floor(2*w/(l+1))
            wl2 = np.floor(wl/2 - 1)

            if l+Wd-1==0:
                b=0
            else:
                b = (self.W - wl) / (l + Wd - 1)
            cenW = np.floor(wl2 + np.arange(0,l+Wd)*b) - wl2 # center coordinates

            if l+Hd-1==0:
                b=0
            else:
                b = (self.H-wl)/(l+Hd-1)
            cenH = np.floor(wl2 + np.arange(0,l+Hd)*b) - wl2 # center coordinates

            for i_ in cenH:
                for j_ in cenW:
                    R = np.array([j_, i_, wl, wl], dtype=np.int)
                    if not min(R[2:]):
                        continue
                    regions.append(R)

        regions = np.asarray(regions)
        return regions

    def forward(self,input_feature):
        self.num_samples=input_feature.shape[0]
        self.num_feature=input_feature.shape[1]
        self.W=input_feature.shape[2]
        self.H=input_feature.shape[3]

        regions=self.get_regions()

        outputs = []

        for roi_idx in range(len(regions)):

            x = regions[roi_idx, 0]
            y = regions[roi_idx, 1]
            w = regions[roi_idx, 2]
            h = regions[roi_idx, 3]

            x1 = int(np.round(x))
            x2 = int(np.round(x1 + h))
            y1 = int(np.round(y))
            y2 = int(np.round(y1 + w))

            x_crop = input_feature[:, :, y1:y2, x1:x2]

            pooled_val=torch.max(x_crop.contiguous().view(self.num_samples,self.num_feature,-1),2)[0]
            outputs.append(pooled_val)

            final_output=outputs[0]
            for item in outputs[1:]:
                final_output+=item
        return final_output

class Mac_Pooling(nn.Module):
    def __init__(self):
        super(Mac_Pooling,self).__init__()
    
    def forward(self,x):
        dim=x.size()
        pool=nn.MaxPool2d(dim[-1])
        x=pool(x)

        return x.view(dim[0],dim[1])

class SPoC_pooling(nn.Module):
    
    def __init__(self):
        super(SPoC_pooling,self).__init__()
    
    def forward(self, x):
        dim=x.size()
        pool=nn.AvgPool2d(dim[-1])
        x=pool(x)

        return x.view(dim[0],dim[1])

def ramac(x, L=3, eps=1e-6, p=1):
    ovr = 0.4 # desired overlap of neighboring regions
    steps = torch.Tensor([2, 3, 4, 5, 6, 7]) # possible regions for the long dimension
    
    W = x.size(3)
    H = x.size(2)
    
    w = min(W, H)
    w2 = math.floor(w/2.0 - 1)
    
    b = (max(H, W)-w)/(steps-1)
    (tmp, idx) = torch.min(torch.abs(((w**2 - w*b)/w**2)-ovr), 0) # steps(idx) regions for long dimension
    
    # region overplus per dimension
    Wd = 0
    Hd = 0
    #print(idx.tolist())
    if H < W:
        Wd = idx.tolist()#[0]
    elif H > W:
        Hd = idx.tolist()#[0]

    v = F.max_pool2d(x, (x.size(-2), x.size(-1)))

    x_min=x.sum(1).min()
    threshold=(x.sum(1)-x_min).pow(p).mean().pow(1/p)+x_min

    # find attention
    tt=(x.sum(1)-threshold>0)
    # caculate weight
    weight=tt.sum().float()/tt.size(-2)/tt.size(-1)
    # ingore
    if weight.data<=1/3.0:
        weight=weight-weight

    v = v / (torch.norm(v, p=2, dim=1, keepdim=True) + eps).expand_as(v) * weight

    for l in range(1, L+1):
        wl = math.floor(2*w/(l+1))
        wl2 = math.floor(wl/2 - 1)
    
        if l+Wd == 1:
            b = 0
        else:
            b = (W-wl)/(l+Wd-1)
        cenW = torch.floor(wl2 + torch.Tensor(range(l-1+Wd+1))*b) - wl2 # center coordinates
        if l+Hd == 1:
            b = 0
        else:
            b = (H-wl)/(l+Hd-1)
        cenH = torch.floor(wl2 + torch.Tensor(range(l-1+Hd+1))*b) - wl2 # center coordinates
        
        for i_ in cenH.tolist():
            for j_ in cenW.tolist():
                if wl == 0:
                    continue
                R = x[:,:,(int(i_)+torch.Tensor(range(int(wl))).long()).tolist(),:]
                R = R[:,:,:,(int(j_)+torch.Tensor(range(int(wl))).long()).tolist()]
                # obtain map
                # tt=(x.sum(1)-x.sum(1).mean()>0)[:,(int(i_)+torch.Tensor(range(int(wl))).long()).tolist(),:][:,:,(int(j_)+torch.Tensor(range(int(wl))).long()).tolist()]
                
                x_min=x.sum(1).min()
                threshold=(x.sum(1)-x_min).pow(p).mean().pow(1/p)+x_min
                # find attention
                tt=(x.sum(1)-threshold>0)[:,(int(i_)+torch.Tensor(range(int(wl))).long()).tolist(),:][:,:,(int(j_)+torch.Tensor(range(int(wl))).long()).tolist()]
                
                vt = F.max_pool2d(R, (R.size(-2), R.size(-1)))
                # caculate each region
                weight=tt.sum().float()/tt.size(-2)/tt.size(-1)
                if weight.data<=1/3.0:
                    weight=weight-weight
                vt = vt / (torch.norm(vt, p=2, dim=1, keepdim=True) + eps).expand_as(vt) * weight
                v += vt
    return v

class Ramac_Pooling(nn.Module):
    
    def __init__(self, L=3, eps=1e-6):
        super(Ramac_Pooling,self).__init__()
        self.L = L
        self.eps = eps
    
    def forward(self, x):
        out = ramac(x, L=self.L, eps=self.eps)
        return out.squeeze(-1).squeeze(-1)

class Grmac_Pooling(nn.Module):
    
    def __init__(self, L=3, eps=1e-6, p=1):
        super(Grmac_Pooling,self).__init__()
        self.L = L
        self.eps = eps
        self.p = p
    
    def forward(self, x):
        out = ramac(x, L=self.L, eps=self.eps, p=self.p)
        return out.squeeze(-1).squeeze(-1)
