import os, sys
import pickle
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

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

# def eval_model_(model, data, device):
#     model.eval()
#     correct = [] 
#     pic = 0
#     with torch.no_grad():
#         for batch_num, (image, label) in enumerate(data):
#             image_train = image.to(device=device)
#             label.to(device=device)
#             pic += len(label)
#             output = model(image_train)
#             output = torch.nn.functional.softmax(output, dim=1)
#             _, _, right = predict(output, label)
#             correct.extend(right)
#     print('acc: {}%'.format(len(correct)/pic * 100))

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

def save_model(model, root, epoch_num, mean_std, eval_flag=True):
    if not os.path.exists(root):
        os.makedirs(root)
    weight = {}
    weight['state_dict'] = model.state_dict()
    weight['mean_std'] = mean_std
    torch.save(weight, os.path.join(root, 'epoch_{}.pth'.format(epoch_num)))


def train(model, train_data, val_data, classes, device, epochs=200):
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=9.99999974738e-05)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=160, gamma=0.1)
    lr_list = [optimizer.state_dict()['param_groups'][0]['lr']]

    info = {}

    for epoch in range(epochs):
        print('current_epoch: {}'.format(epoch))
        '''
        if epoch % 5 ==0 and epoch != 0 and epoch < 2000:
            for p in optimizer.param_groups:
                p['lr'] *= 0.9
                lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
        '''
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
            loss = loss1 + 0.3 * (loss2 + loss3)

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
        save_model(model,'inception_bn', epoch, msd)
        print('='*41)
    info['lr_list'] = lr_list

    with open(os.path.join('./', 'lr.pkl'), 'wb') as f:
        pickle.dump(info, f)

if __name__ == "__main__":
    try:
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
    # model = googlenet(num_classes=len(classes))
    model.apply(init_weights)

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    model.to(device=device)
    train(model, train_data, val_data, train_dataset.classes, device)
