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

#root = sys.argv[1]
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

classes = ('fangkuaibubaoman','fangkuaiqipao', 'ok')

train_dataset = datasets.ImageFolder('train',transform=transform)
print("train dataset label: ", train_dataset.class_to_idx)
print("train has {} pictures".format(len(train_dataset)))
train_data = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = datasets.ImageFolder('test',transform=transform)
print("test dataset label: ", test_dataset.class_to_idx)
print("test has {} pictures".format(len(test_dataset)))
test_data = DataLoader(test_dataset, batch_size=32, shuffle=True)
val_dataset = datasets.ImageFolder('val',transform=transform)
print("validation dataset label: ", val_dataset.class_to_idx)
print("validation has {} pictures".format(len(val_dataset)))
val_data = DataLoader(val_dataset, batch_size=32, shuffle=True)

def init_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.xavier_normal_(module.weight.data)
        
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0.20000000298)

def eval_model_(model, data, device):
    model.eval()
    correct = [] 
    pic = 0
    with torch.no_grad():
        for batch_num, (image, label) in enumerate(data):
            image_train = image.to(device=device)
            label.to(device=device)
            pic += len(label)
            output = model(image_train)
            output = torch.nn.functional.softmax(output, dim=1)
            _, _, right = predict(output, label)
            correct.extend(right)
    print('acc: {}%'.format(len(correct)/pic * 100))

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
        try:
            print('| {} | acc: {} | recall: {} |'.format(_class, info[_class]['TP']/(info[_class]['TP'] + info[_class]['FP']), info[_class]['TP']/(info[_class]['TP'] + info[_class]['FN'])))
        except ZeroDivisionError as e:
            print(_class + " is not ready for checking acc and recall")
            print(_class + ' TP: {}'.format(info[_class]['TP']))
            print(_class + " FP: {}".format(info[_class]['FP']))
            print(_class + ' TN: {}'.format(info[_class]['TN']))
            print(_class + ' FN: {}'.format(info[_class]['FN']))
    return info

def load_model(weight, device):
    model = torchvision.models.googlenet(num_classes=len(classes))
    model.to(device=device).eval()
    model.load_state_dict(torch.load(weight))
    return model

def save_model(model, root, epoch_num, eval_flag=True):
    if not os.path.exists(root):
        os.makedirs(root)
    torch.save(model.state_dict(),os.path.join(root, 'epoch_{}.pth'.format(epoch_num)))

model = torchvision.models.googlenet(num_classes=len(classes))
# model = googlenet(num_classes=len(classes))
model.apply(init_weights)

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'
# model.load_state_dict(torch.load('work_dirs/inception_bn2/epoch_3999.pth'))
model.to(device=device)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=9.99999974738e-05)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=160, gamma=0.1)
lr_list = [optimizer.state_dict()['param_groups'][0]['lr']]

info = {}

for epoch in range(4000):
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
        loss1 = nn.functional.cross_entropy(main_out, label_train)
        loss2 = nn.functional.cross_entropy(aux1_out, label_train)
        loss3 = nn.functional.cross_entropy(aux2_out, label_train)
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
    info = eval_model(model, val_data, device)
    save_model(model,'inception_bn', epoch)
    print('='*41)
info['lr_list'] = lr_list

with open(os.path.join('./', 'lr.pkl'), 'wb') as f:
    pickle.dump(info, f)
