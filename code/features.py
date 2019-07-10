import os
import argparse
import numpy as np
from PIL import Image
import shutil

from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import pickle
import warnings
from tqdm import tqdm
from pooling import *
from training_dataset import retrieval_dataset
import net2

transform_480 = transforms.Compose([
    transforms.Resize((480,480)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

model_path={
    'densenet201':'./pretrained/densenet201.t7',
    'seresnet152':'./pretrained/seresnet152.t7',
}
feature_length={
    'densenet201':1920,
    'seresnet152':2048
}

if __name__ == "__main__":
    name_list=os.listdir('../dataset_clean')
    name_list.sort()
    mode='cuda' # or 'cpu'

    for model_name in ['seresnet152','densenet201']:
        model=net2.__dict__[model_name](model_path[model_name])
        if mode=='cuda':
            model=model.cuda()
            model = torch.nn.DataParallel(model)
            cudnn.benchmark = True

        dataset = retrieval_dataset('../dataset_clean',transform=transform_480)
        testloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False, num_workers=4)
        
        feat_dict={
            'rmac':torch.empty(len(name_list),feature_length[model_name]),
            'ramac':torch.empty(len(name_list),feature_length[model_name]),
            'Grmac':torch.empty(len(name_list),feature_length[model_name]),
            'SPoC':torch.empty(len(name_list),feature_length[model_name]),
            'Mac':torch.empty(len(name_list),feature_length[model_name])
        }
        print(feat_dict['rmac'].size())
        img_list=[]
        model.eval()
        with torch.no_grad():
            for i, (inputs, names) in tqdm(enumerate(testloader)):
                inputs = inputs.to(mode)
                feature_rmac,feature_ramac,feature_Grmac,feature_Mac,feature_SPoC, = model(inputs)
                # print(features.size())
                
                feat_dict['rmac'][i*100:i*100+len(names),:]=feature_rmac.cpu()
                feat_dict['ramac'][i*100:i*100+len(names),:]=feature_ramac.cpu()
                feat_dict['Grmac'][i*100:i*100+len(names),:]=feature_Grmac.cpu()
                feat_dict['SPoC'][i*100:i*100+len(names),:]=feature_SPoC.cpu()
                feat_dict['Mac'][i*100:i*100+len(names),:]=feature_Mac.cpu()
                
                assert name_list[i*100:i*100+len(names)]==list(names)
                img_list.extend(names)
        
        with open("./feature/feat_{}.pkl".format(model_name), "wb") as file_to_save:
            pickle.dump(
                {
                'name':img_list,
                'rmac':feat_dict['rmac'].half().numpy(),
                'ramac':feat_dict['ramac'].half().numpy(),
                'Grmac':feat_dict['Grmac'].half().numpy(),
                'SPoC':feat_dict['SPoC'].half().numpy(),
                'Mac':feat_dict['Mac'].half().numpy()
                    }, 
                file_to_save, 
                -1
                )