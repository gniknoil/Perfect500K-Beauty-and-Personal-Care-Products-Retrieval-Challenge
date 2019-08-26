#  Perfect Half Million Beauty Product Image Recognition Challenge


This is our solution for [Perfect Half Million Beauty Product Image Recognition Challenge
](https://challenge2019.perfectcorp.com/index.html), which obtained the 1st place (USTC_NELSLIP) with MAP@7 0.4086


## Dependency

- python3.5
- pytorch(1.1.0)
- torchvision
- [pretrainedmodels](https://github.com/Cadene/pretrained-models.pytorch)
- tqdm
- PIL

## Preparation

  ### For Image Database

  1.Download all the images of dataset recorded in [csvfile](https://drive.google.com/open?id=1ZWnaLSw0K8VDmKJ8MgZZCPlS8IEZkbOO)

  2.Transform all the images to square form using 255 padding

  3.Put all the images in the folder data_clean

  ### For Query Images

  Query images are put in the folder search/ori_images
  
  ### For models
  
  [SEResnet152](https://drive.google.com/open?id=1IL4eNa-4_WTCiQTfjmtTiyjco_twFX3I)
  [Densenet201](https://drive.google.com/open?id=1wZdnfXLoz6O9s-0ONyG2bDC-g9j1dVdX)
  
  Download these two models and put them in the folder /code/pretrained

  ### Usage

  1. Extract features of images in database(Perfect-500K)

    $ python features.py

  2. Conduct image retrieval

    $ python search.py
  
### Note

All the models are all pretrained on the Imagenet. They can be downloaded from module [pretrainedmodels](https://github.com/Cadene/pretrained-models.pytorch). 

