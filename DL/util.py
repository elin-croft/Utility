import os, sys
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets

def predict(input, labels:torch.Tensor):
    result = torch.argmax(input, dim=1).tolist()
    correct = []
    label = labels.tolist()
    for det, gt in zip(result, label):
        if det == gt:
            correct.append(det)
    # print('acc: {}%'.format(correct/len(result) * 100))

    return result, label, correct

def load_dataset(root, transform,
                batch_size=32, shuffle=True, 
                dataset_type='folder', 
                *args, **kwargs):
    """
    param
    dataset_type: str
        should be voc , coco, cifar, minst or folder
    
    """
    if dataset_type == 'folder':
        dataset = datasets.ImageFolder(root, transform=transform)

    elif dataset_type == 'voc':
        year = kwargs['year']
        image_set = kwargs['image_set']
        dataset = datasets.VOCDetection(root, year=year, image_set=image_set, transform=transform)
    elif dataset_type == 'coco':
        annfile = kwargs['annfile']
        type=kwargs['type']
        if type == 'detect':
            dataset = datasets.CocoDetection(root, annFile=annfile, transform=transform)
        elif type == 'caption':
            dataset = datasets.CocoCaptions(root, annFile=annfile, transform=transform)

    data = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return data, dataset

def init_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.xavier_normal_(module.weight.data)
        
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0.20000000298)

def eval_model(model, data, classes, device):
    info = {}
    model.eval()
    results = []
    correct = [] 
    labels = []
    pic = 0
    with torch.no_grad():
        for batch_num, (image, label) in enumerate(data):
            image_train = image.to(device=device)
            pic += len(label)
            output = model(image_train)
            output = torch.nn.functional.softmax(output, dim=1)
            result, label_list, right = predict(output, label)
            correct.extend(right)
            results.extend(result)
            labels.extend(label_list)
    
    print('total acc: {}%'.format(len(correct)/pic * 100))
    for index, _class in enumerate(classes):
        info.setdefault(_class, {})['TP'] = correct.count(index)
        info[_class]['FP'] = results.count(index) - correct.count(index)
        nagitive = len(results) - results.count(index)
        #print('nagitive {}'.format(nagitive))
        info[_class]['TN'] = len(correct) - correct.count(index)
        info[_class]['FN'] = nagitive - len(correct) + correct.count(index)
        #print(info[_class]['TP'])
        #print(info[_class]['FP'])
        #print(info[_class]['TN'])
        #print(info[_class]['FN'])
        try:
            print('| {} | acc: {} | recall: {} |'.format(_class, info[_class]['TP']/(info[_class]['TP'] + info[_class]['FP']), info[_class]['TP']/(info[_class]['TP'] + info[_class]['FN'])))
        except ZeroDivisionError as e:
            print(_class + " is not ready for checking acc and recall")
            print(_class + ' TP: {}'.format(info[_class]['TP']))
            print(_class + " FP: {}".format(info[_class]['FP']))
            print(_class + ' TN: {}'.format(info[_class]['TN']))
            print(_class + ' FN: {}'.format(info[_class]['FN']))
    return info

def load_model(model, weight, device):
    model.to(device=device).eval()
    model.load_state_dict(torch.load(weight))
    return model

def save_model(model, root, epoch_num, flavour='normal', *args, **kwagrs):
    """
    save model
    Parameters
    flavour: str
        normal: save a model in normal way
        compressed: save a model with other parameters
                    but mean and std is needed
    """
    if not os.path.exists(root):
        os.makedirs(root)
    if flavour == 'compressed':
        weight = {}
        weight['state_dict'] = model.state_dict()
        for k, v in kwagrs.items():
            weight[k] = v
        torch.save(weight, os.path.join(root, 'epoch_{}.pth'.format(epoch_num)))
    elif flavour == 'normal':
        torch.save(model.state_dict(), os.path.join(root, 'epoch_{}.pth'.format(epoch_num)))