import numpy as np
import cv2
import os, sys
from calculate_size import calculate_size
def calculate(root):
    h, w = calculate_size(root)
    imgs=[]
    means=[]
    stds=[]
    files = os.listdir(root)
    for pic in files:
        if not pic.startswith('.') and pic.endswith('.jpg'):
            img = cv2.imread(os.path.join(root, pic))
            img = cv2.resize(img, (w, h))
            cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = img[:, :, :, np.newaxis]
            imgs.append(img)
    nparr = np.concatenate(imgs, axis=3)
    for i in range(3):
        pixel = nparr[:,:,i,:].ravel()
        #means.append(np.mean(pixel))
        stds.append(np.std(pixel))
    print("mean: ", means)
    print("stds: ", stds)
if __name__ == "__main__":
    if len(sys.argv) > 1:
        root = sys.argv[1]
    else:
        root='./'
    calculate(root)
