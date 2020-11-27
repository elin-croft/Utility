import os, sys
import pickle
import cv2
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

classes = ('maoci', 'ok1', 'ok2', 'ok3', 'shangzhibubaoman')

def load_model(models, weight, device):
    
    model.to(device=device).eval()
    model.load_state_dict(torch.load(weight))
    return model

def test(model, weight, test_data, device):
    model = load_model(model, weight, device)
    output = model(test_data)
    label = torch.argmax(output).item()
    print(type(label))
    print(classes[label])

if __name__ == '__main__':
    f = open('mean_std.pkl', 'rb')
    msd = pickle.load(f)
    f.close()
    #mean = tuple(map(lambda x:round(x ,1), msd['mean']))
    #std = tuple(map(lambda x:round(x ,1), msd['std']))
    mean = msd['mean']
    std = msd['std']

    print("mean={}".format(mean))
    print("std={}".format(std))

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])

    root = sys.argv[1]
    test_img = cv2.imread(root)
    print("test dataset label: ", classes)
    img = transform(test_img).unsqueeze(0).to(device='cuda:0') 
    model = torchvision.models.googlenet(num_classes=len(classes))
    weight = 'work_dirs/best/epoch_3999.pth'   
    test(model, weight, img, 'cuda:0')
