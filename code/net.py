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
    def __init__(self,model_path,feature_name):
        super(seresnet152, self).__init__()
        se152 = se_resnet152(pretrained=None)
        checkpoint=torch.load(model_path)
        se152.load_state_dict(checkpoint)
        self.norm=L2N()
        self.backbone=nn.Sequential(*list(se152.children())[:-2])
        self.feature_name=feature_name
        if self.feature_name=='rmac':
            self.rmac=Rmac_Pooling()
        if self.feature_name=='ramac':
            self.ramac=Ramac_Pooling()
        if self.feature_name=='Grmac':
            self.Grmac=Grmac_Pooling()
        if self.feature_name=='Mac':
            self.Mac=Mac_Pooling()
        if self.feature_name=='SPoc':
            self.SPoc=SPoC_pooling()

    def forward(self,data):
        feature=self.backbone(data)
        if self.feature_name=='rmac':
            feature=self.rmac(feature)
        if self.feature_name=='ramac':
            feature=self.ramac(feature)
        if self.feature_name=='Grmac':
            feature=self.Grmac(feature)
        if self.feature_name=='Mac':
            feature=self.Mac(feature)
        if self.feature_name=='SPoc':
            feature=self.SPoc(feature)
        feature=self.norm(feature)
        return feature

class densenet201(nn.Module):
    def __init__(self,model_path,feature_name):
        super(densenet201, self).__init__()
        dense201 = models.densenet201()
        checkpoint=torch.load(model_path)
        dense201.load_state_dict(checkpoint)
        self.norm=L2N()
        self.backbone=nn.Sequential(*list(dense201.children())[:-1])
        self.feature_name=feature_name
        if self.feature_name=='rmac':
            self.rmac=Rmac_Pooling()
        if self.feature_name=='ramac':
            self.ramac=Ramac_Pooling()
        if self.feature_name=='Grmac':
            self.Grmac=Grmac_Pooling()
        if self.feature_name=='Mac':
            self.Mac=Mac_Pooling()
        if self.feature_name=='SPoc':
            self.SPoc=SPoC_pooling()

    def forward(self,data):
        feature=self.backbone(data)
        if self.feature_name=='rmac':
            feature=self.rmac(feature)
        if self.feature_name=='ramac':
            feature=self.ramac(feature)
        if self.feature_name=='Grmac':
            feature=self.Grmac(feature)
        if self.feature_name=='Mac':
            feature=self.Mac(feature)
        if self.feature_name=='SPoc':
            feature=self.SPoc(feature)
        feature=self.norm(feature)
        return feature