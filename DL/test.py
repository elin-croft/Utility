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

def eval_model(model, data, device):
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

def test(model, weight, test_data, device):
    model = load_model(model, weight, device)
    eval_model(model, test_data, device)

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
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])

    root = sys.argv[1]
    test_dataset = datasets.ImageFolder(os.path.join(root, 'test'),transform=transform)
    classes = tuple(test_dataset.classes)
    print("test dataset label: ", test_dataset.class_to_idx)
    test_data = DataLoader(test_dataset, batch_size=32, shuffle=True)
    
    model = torchvision.models.googlenet(num_classes=len(classes))
    weight = 'work_dirs/inception_bn/epoch_200.pth'   
    model = load_model(model, weight, 'cuda:0')
    eval_model(model, test_data, 'cuda:0')

