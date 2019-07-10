import os
import pickle
from PIL import Image
import torch
import torch.utils.data as data
import numpy as np

def conversion(img):
    #img: PIL format
    img=np.array(img)
    h,w,c=img.shape
    if h>w:
        img2=np.ones((h,h,c),dtype='uint8')*255
        start=(h-w)//2
        img2[:,start:start+w,:]=img
    elif w>h:
        img2=np.ones((w,w,c),dtype='uint8')*255
        start=(w-h)//2
        img2[start:start+h,:,:]=img
    else:
        img2=img
    img2=Image.fromarray(img2)
    return img2

class retrieval_dataset(data.Dataset):
    def __init__(self,root_path,transform = None,crop_aug=False):
        self.root=root_path
        self.image_list=os.listdir(self.root)
        self.image_list.sort()
        self.transform=transform
    
    def __getitem__(self,idx):
        image=Image.open(os.path.join(self.root,self.image_list[idx])).convert('RGB')
        image=conversion(image)
        output_imgs=self.transform(image)
        return output_imgs, self.image_list[idx]

    def __len__(self):
        return len(self.image_list)
