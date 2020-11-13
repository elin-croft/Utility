import os, sys
import random
import shutil

def get_scripe(total, k):
    scripe = random.sample(range(total), k)
    return scripe

def split_dataset(root):
    files = os.listdir(root)
    data_files = [os.path.join(root, i) for i in files if i.endswith('.jpg')]
    file_len = len(data_files)
    test_len = int(file_len * 0.2)
    val_len = int((file_len - test_len) * 0.2)
    train_len = int(file_len - test_len - val_len)
    print("total {}; train {}; test {}; val {}".format(file_len, train_len, test_len, val_len))
    test = get_scripe(file_len, test_len)
    test_files = [data_files[i] for i in test]

    train_val_files = [i for i in data_files if i not in test_files]
    val = get_scripe(file_len - test_len, val_len)
    val_files = [train_val_files[i] for i in val]
    train_files = [i for i in train_val_files if i not in val_files]
    return train_files, test_files, val_files

def check_dataset(train, test, val):
    total = len(train) + len(test) + len(val)
    ttintersection = [i for i in train if i in test]
    tvintersection = [i for i in train if i in val]
    tevintersection = [i for i in test if i in val]
    if len(ttintersection) > 0:
        print('train and test set has intersection')
        return -1

    if len(tvintersection) > 0:
        print('train and val set has intersection')
        return 2

    if len(tevintersection) > 0:
        print('test and val set has intersection')
        return -3

    print("proportion: test/total: {}; val/train_val: {}".format(len(test) / total, len(val) / (len(train) + len(val))))

    return 1
                                                                                                                             
def move_data(root, data):
    # print(root)
    if not os.path.exists(root):
        os.makedirs(root)
    for item in data:
        filename = os.path.split(item)[-1]
        # print("cp {} {}".format(item, os.path.join(root, filename)))
        os.system("cp {} {}".format(item, os.path.join(root, filename)))

if __name__ == "__main__":
    root = sys.argv[1]
    train, test, val = split_dataset(root)
    check_dataset(train, test, val)
    if root.endswith('/'):
       root = root[0:-1]
    sub_root, label = os.path.split(root)
    print(sub_root, label)
    move_data(os.path.join('dataset/folder/path', sub_root, 'train', label), train)
    move_data(os.path.join('dataset/folder/path', sub_root, 'test', label), test)
    move_data(os.path.join('dataset/folder/path', sub_root, 'val', label), val)
                                                           