import cv2
import numpy as np
import os, sys
import math

def calculate_size(root):
    info={'h':[], 'w':[]}
    pic_files = os.listdir(root)
    for pic in pic_files:
        if pic.endswith('.jpg'):
            img=cv2.imread(os.path.join(root, pic))
            info['h'].append(img.shape[0])
            info['w'].append(img.shape[1])

    height = np.array(info['h'])
    width = np.array(info['w'])
    weight_h = height / np.sum(height)
    weight_w = width / np.sum(width)
    reasonable_h = math.ceil(np.sum(height * weight_h))
    reasonable_w = math.ceil(np.sum(width * weight_w))
    print(reasonable_h, ' ', reasonable_w)
    return int(reasonable_h), int(reasonable_w)

if __name__ == "__main__":
    root = './'
    calculate_size(root)
