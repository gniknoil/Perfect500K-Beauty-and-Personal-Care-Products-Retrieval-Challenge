import sys
import os
import csv
from tqdm import tqdm
import numpy as np
import pickle
import warnings

import torch
import torchvision
import torchvision.models as models

from training_dataset import retrieval_dataset
from pooling import *
from features import transform_480
from net import seresnet152, densenet201


def load_feature(feat_name):
    with open(feat_name, "rb") as file_to_read:
        feature=pickle.load(file_to_read)
    name=feature['name']
    return name,feature

if __name__ == "__main__":
    test_image_path=sys.argv[1]
    result_path=sys.argv[2]

    se_path='./pretrained/seresnet152.t7'
    dense_path='./pretrained/densenet201.t7'
    se_feature_path='./feature/feat_seresnet152.pkl'
    dense_feature_path='./feature/feat_dense201.pkl'

    name_list,dense_feature=load_feature(dense_feature_path)
    name_list,senet_feature=load_feature(se_feature_path)

    feature={'dense201':dense_feature,'seresnet152':senet_feature}
    feat_type={'dense201':['Mac'],'seresnet152':['Grmac']}
    weight={
        'dense201':{'Mac':1,'rmac':0,'ramac':0,'Grmac':0},
        'seresnet152':{'Mac':0,'rmac':0,'ramac':0,'Grmac':1}
    }
    dim_feature={
        'dense201':1920,
        'seresnet152':2048
    }
    batch_size=20

    similarity=torch.zeros(len(os.listdir(test_image_path)),len(name_list))
    print(similarity.size())

    for model_name in ['dense201','seresnet152']:
        feature_model=feature[model_name]
        for item in feat_type[model_name]:
            if model_name == 'seresnet152':
                model=seresnet152(se_path,item)
            elif model_name == 'dense201':
                model=densenet201(dense_path,item)
            else:
                pass

            feat_reserved=feature_model[item]

            dataset = retrieval_dataset(test_image_path,transform=transform_480)
            testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

            model=model.cpu()
            model.eval()
            query=torch.empty(len(os.listdir(test_image_path)),dim_feature[model_name])
            name_test=[]
            with torch.no_grad():
                for i, (inputs, names) in tqdm(enumerate(testloader)):
                    query[i*batch_size:i*batch_size+len(names)] = model(inputs).cpu()
                    name_test.extend(names)

            feat_reserved=torch.Tensor(feat_reserved).transpose(1,0)
            query=torch.Tensor(query)
            
            similarity+=torch.matmul(query,feat_reserved)*weight[model_name][item]

    _, predicted = similarity.topk(7)
    predicted=predicted.tolist()
    dict_result=dict(zip(name_test,predicted))

    #saving csv
    img_results=[]
    name_test.sort()
    for name in name_test:
        temp=[name.split('.')[0]]
        for idx in dict_result[name]:
            temp.append(name_list[idx].split('.')[0])
        img_results.append(temp)
    print('saving')
    out = open(result_path,'w')
    csv_write = csv.writer(out)
    csv_write.writerows(img_results)
