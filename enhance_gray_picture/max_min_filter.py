import numpy as np
from im2col_cython import im2col
import cv2

def maxfilter(image, kernel):

    assert np.ndim(image) in (2, 3)
    if np.ndim(image) == 2:
        h, w = image.shape
        c = 1
    else:
        image = np.transpose(image, (2, 0, 1))
        c, h, w = image.shape
    # filtered_array = -np.ones((image.shape))
    kernel_size = kernel
    assert kernel_size % 2 != 0
    pading = kernel // 2
    a = np.pad(image, ((pading, ), (pading, )), 'constant', constant_values=(-1, -1))
    
    image_col = im2col(a, c, h, w, kernel)
    result = np.amax(image_col, axis=1).reshape(h, w)
    return result

def minfilter(image, kernel):

    assert np.ndim(image) in (2, 3)
    if np.ndim(image) == 2:
        h, w = image.shape
        c = 1
    else:
        image = np.transpose(image, (2, 0, 1))
        c, h, w = image.shape
    # filtered_array = -np.ones((image.shape))
    kernel_size = kernel
    assert kernel_size % 2 != 0
    pading = kernel // 2
    a = np.pad(image, ((pading, ), (pading, )), 'constant', constant_values=(300, 300))
    
    image_col = im2col(a, c, h, w, kernel)
    result = np.amin(image_col, axis=1).reshape(h, w)
    return result

if __name__ == "__main__":
    img = cv2.imread('images/elsa.jpg',0)
    import time
    t0 = time.time()
    out = maxfilter(img, 31)
    print('takes {}s'.format(time.time() - t0))
    print(out.shape)