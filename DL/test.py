import os, sys
import pickle
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

# classes = ('fangkuaibubaoman','fangkuaiqipao', 'ok')
def predict(input, labels:torch.Tensor):
    result = torch.argmax(input, dim=1).tolist()
    correct = []
    label = labels.tolist()
    for det, gt in zip(result, label):
        if det == gt:
            correct.append(det)
    # print('acc: {}%'.format(correct/len(result) * 100))

    return result, label, correct
#root = sys.argv[1]

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
        print('| {} | acc: {} | recall: {} |'.format(_class, info[_class]['TP']/(info[_class]['TP'] + info[_class]['FP']), info[_class]['TP']/(info[_class]['TP'] + info[_class]['FN'])))

def load_model(models, weight, device):
    
    model.to(device=device).eval()
    model.load_state_dict(torch.load(weight))
    return model

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

    return data, dataset.classes, dataset.class_to_idx
    

def test(model, weight, test_data, classes,device):
    model = load_model(model, weight, device)
    eval_model(model, test_data, classes,device)

if __name__ == '__main__':
    # load mean and std from pkl file
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

    root = sys.argv[1]
    test_dataset, classes, class_to_idx = load_dataset(os.path.join(root, 'test'), transform, 32, True)
    classes = tuple(classes)
    print("test dataset label: ", class_to_idx)
    test_data = DataLoader(test_dataset, batch_size=32, shuffle=True)
    
    model = torchvision.models.googlenet(num_classes=len(classes))
    weight = 'work_dirs/inception_bn/epoch_200.pth'
    test(model, weight, test_data, classes, 'cuda:0')

