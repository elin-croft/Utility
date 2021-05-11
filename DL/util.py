import os, sys
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets

def predict(input, labels:torch.Tensor):
    """
    Parameters
    -----------

    input: Tensor shape (n, k) where n is batch number and k is number of classes
        output of nerual network
    
    labels: Tensor shape (n, )
        true classes of batch 
    
    Return
    ----------
    result: list
        inference results

    label: list
        true classes

    correct: list
        correct results

    """
    result = torch.argmax(input, dim=1).tolist()
    correct = []
    label = labels.tolist()
    for det, gt in zip(result, label):
        if det == gt:
            correct.append(det)

    return result, label, correct

def load_dataset(root, transform,
                 batch_size=32, shuffle=True, 
                 dataset_type='folder', 
                 *args, **kwargs):
    """
    Parameters
    -----------

    dataset_type: str
        should be voc , coco, cifar, minst or folder
    
    Return
    ----------
    data: Dataloader

    dataset: torchvision.dataset

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

def init_model(model, func):
    """
    Parameters
    -----------

    model: nn.Module
        your model
    
    func: functional
        your initail function
    """
    model.apply(func)

def eval_model(model, data, classes, device='cpu'):
    """
    model: nn.Module
    
    data: Dataloader
    """
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
        info[_class]['TN'] = len(correct) - correct.count(index)
        info[_class]['FN'] = nagitive - len(correct) + correct.count(index)
        try:
            print('| {} | acc: {} | recall: {} |'.format(_class, info[_class]['TP']/(info[_class]['TP'] 
                + info[_class]['FP']), info[_class]['TP']/(info[_class]['TP'] 
                + info[_class]['FN'])))
        except ZeroDivisionError as e:
            print(_class + " is not ready for checking acc and recall")
            print(_class + ' TP: {}'.format(info[_class]['TP']))
            print(_class + " FP: {}".format(info[_class]['FP']))
            print(_class + ' TN: {}'.format(info[_class]['TN']))
            print(_class + ' FN: {}'.format(info[_class]['FN']))
    return info

def load_model(model, weight, device, flavour='normal'):
    """
    Parameters
    -----------

    model: nn.Module

    weight: str
        path to you .pth or .pt file
    
    device:str
        set your model to cpu or gpu

    flavour:str
        if your .pth has other parameters like mean or std or classer you should choose compressed
        
        e.g.
        pthFile = torch.load(weight)
        state_dict = pthFile['state_dict']
        mean = stat_dict = pthFile=['mean']

    
    Return
    --------

    compressed_param:ditc
        all parameters that is compressed in pth file execpt state_dict

    """
    assert 'cpu' in device or "cuda" in device
    model.to(device=device).eval()
    model_data = torch.load(weight)
    compressed_param = None
    if flavour == 'normal':
        model.load_state_dict(model_data)
    elif flavour == 'compressed':
        try:
            state_dict = model_data['state_dict']
        except KeyError as e:
            print("your weight file doesn't have statc dict which is the real weight")
        model.load_state_dict(state_dict)
        model_data.pop('state_dict')
        compressed_param = {k: v for k, v in model_data.items()}

    return model, compressed_param

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