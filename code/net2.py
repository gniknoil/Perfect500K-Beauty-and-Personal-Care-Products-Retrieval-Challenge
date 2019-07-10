import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from pooling import *
from pretrainedmodels import se_resnet152

__all__=['L2N','seresnet152','densenet201']

#---------------feature extraction------------------#

class L2N(nn.Module):

    def __init__(self, eps=1e-6):
        super(L2N,self).__init__()
        self.eps = eps

    def forward(self, x):
        return x / (torch.norm(x, p=2, dim=1, keepdim=True) + self.eps).expand_as(x)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'eps=' + str(self.eps) + ')'

class seresnet152(nn.Module):
    def __init__(self,model_path):
        super(seresnet152, self).__init__()
        se152 = se_resnet152(pretrained=None)
        checkpoint=torch.load(model_path)
        se152.load_state_dict(checkpoint)
        self.norm=L2N()
        self.backbone=nn.Sequential(*list(se152.children())[:-2])
        self.rmac=Rmac_Pooling()
        self.ramac=Ramac_Pooling()
        self.Grmac=Grmac_Pooling(p=3.5)
        self.Mac=Mac_Pooling()
        self.SPoC=SPoC_pooling()

    def forward(self,data):
        feature=self.backbone(data)
        feature_rmac=self.norm(self.rmac(feature))
        feature_ramac=self.norm(self.ramac(feature))
        feature_Grmac=self.norm(self.Grmac(feature))
        feature_Mac=self.norm(self.Mac(feature))
        feature_SPoC=self.norm(self.SPoC(feature))
        return feature_rmac,feature_ramac,feature_Grmac,feature_Mac,feature_SPoC

class densenet201(nn.Module):
    def __init__(self,model_path):
        super(densenet201, self).__init__()
        dense201 = models.densenet201()
        checkpoint=torch.load(model_path)
        dense201.load_state_dict(checkpoint)
        self.norm=L2N()
        self.backbone=nn.Sequential(*list(dense201.children())[:-1])
        self.rmac=Rmac_Pooling()
        self.ramac=Ramac_Pooling()
        self.Grmac=Grmac_Pooling(p=3.5)
        self.Mac=Mac_Pooling()
        self.SPoC=SPoC_pooling()

    def forward(self,data):
        feature=self.backbone(data)
        feature_rmac=self.norm(self.rmac(feature))
        feature_ramac=self.norm(self.ramac(feature))
        feature_Grmac=self.norm(self.Grmac(feature))
        feature_Mac=self.norm(self.Mac(feature))
        feature_SPoC=self.norm(self.SPoC(feature))
        return feature_rmac,feature_ramac,feature_Grmac,feature_Mac,feature_SPoC
