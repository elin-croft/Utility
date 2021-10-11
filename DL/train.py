import os, sys
import pickle
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from util import *

def init_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.xavier_normal_(module.weight.data)
        
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0.20000000298)

def train(model, train_data, val_data, classes, device, 
          save_flavour='normal', epochs=200,
          **kwargs):
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=9.99999974738e-05)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=160, gamma=0.1)
    lr_list = [optimizer.state_dict()['param_groups'][0]['lr']]

    info = {}

    for epoch in range(epochs):
        print('current_epoch: {}'.format(epoch))
        lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
        for batch, (image, label) in enumerate(train_data):
            model.train()
            image_train = image.to(device=device)
            label_train = label.to(device=device)
            output, aux1, aux2 = model(image_train)
            main_out = F.softmax(output, dim=1)
            aux1_out = F.softmax(aux1, dim=1)
            aux2_out = F.softmax(aux2, dim=1)
            regular = 0
            for param in model.parameters():
                regular += torch.sum(param.pow(2))
            loss1 = F.cross_entropy(main_out, label_train)
            loss2 = F.cross_entropy(aux1_out, label_train)
            loss3 = F.cross_entropy(aux2_out, label_train)
            loss = loss1 + 0.3 * (loss2 + loss3) + 0.5 * regular

            # predict(output, label)
            if batch % 10 == 0:
                print('loss: {}'.format(loss.item()))
                _, _, correct = predict(output, label)
                pic = len(label)
                print('acc: {}%'.format(len(correct)/pic * 100))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
        if epoch <= 640:
            scheduler.step()
            #eval_model(model, val_data, device)
            #save_model(model,'work_dirs', epoch)

        print('='*41)
        print('information')
        print('epoch: {}'.format(epoch))
        print('lr: ', lr_list[-1])
        print("testing on val dataset")
        info = eval_model(model, val_data, classes, device)
        save_model(model,'inception_bn', epoch, flavour=save_flavour, **kwargs)
        print('='*41)
    info['lr_list'] = lr_list

    with open(os.path.join('./', 'lr.pkl'), 'wb') as f:
        pickle.dump(info, f)

if __name__ == "__main__":
    try:
        f = open('mean_std.pkl', 'rb')
        msd = pickle.load(f)
        f.close()
        mean = msd['mean']
        std = msd['std']

        print("mean={}".format(mean))
        print("std={}".format(std))
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    except FileNotFoundError as e:
        print(e)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    
    train_data, train_dataset, = load_dataset('train', transform)
    print("train dataset label: ", train_dataset.class_to_idx)
    print("train has {} pictures".format(len(train_dataset)))
    val_data, val_dataset = load_dataset('val', transform)
    print("validation dataset label: ", val_dataset.class_to_idx)
    print("validation has {} pictures".format(len(val_dataset)))

    model = torchvision.models.googlenet(num_classes=len(train_dataset.classes))
    init_model(model, init_weights)

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    model.to(device=device)
    train(model, train_data, val_data, 
        train_dataset.classes, device,
        save_flavour='compressed', mean_std=msd)
