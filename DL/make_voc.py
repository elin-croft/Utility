import os, sys
import random

root = ""
if len(sys.argv) > 1:
    root = sys.argv[1]
trainval_percent = 0.8
train_percent = 0.8
xmlfilepath = 'Annotations'
txtsavepath = 'ImageSets/Main'
total_xml = os.listdir(os.path.join(root, xmlfilepath))

num=len(total_xml)
list=range(num)
tv=int(num*trainval_percent)
tr=int(tv*train_percent)
print('total: ', num)
print('tv: ',tv)
print('tr: ', tr)
trainval= random.sample(list,tv)
train=random.sample(trainval,tr)

ftrainval = open(os.path.join(root, 'ImageSets/Main/trainval.txt'), 'w')
ftest = open(os.path.join(root,'ImageSets/Main/test.txt'), 'w')
ftrain = open(os.path.join(root, 'ImageSets/Main/train.txt'), 'w')
fval = open(os.path.join(root, 'ImageSets/Main/val.txt'), 'w')
for i  in list:
    name=total_xml[i][:-4]+'\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest .close()

