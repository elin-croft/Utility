import argparse
import cv2
import numpy as np
import max_min_filter

def parse_args(params:list=None):
    parser = argparse.ArgumentParser(description="paramters")
    parser.add_argument('-f', '--file', dest='filename',
                        default='./images/elsa.jpg', type=str,
                        help='file name should be something like path/to/you/file.jpg ' \
                            'default set is elsa.jpg')
    parser.add_argument('-c', '--choose', dest='mode',
                        default=0, type=int,
                        help="choose mode between max_min:0 and min_max:1\
                            default set is 0")
    parser.add_argument('--size1', '--filter-size1', dest='size1',
                        default=21, type=int,
                        help="filter size of the first filter which should be an odd number\
                        default set is 21")
    parser.add_argument('--size2', '--filter-size2', dest='size2',
                        default=None, type=int,
                        help="filter size of the second filter. \
                            if you don't set this, size2 will equal to size1\
                            default set is None")
    parser.add_argument('--show', dest='show_flag', 
                        default=False, type=bool,
                        help='show result on your screen\
                            deafult set is False')
    parser.add_argument('--output-name', dest='dst', 
                        default='images/result.jpg', type=str,
                        help='save result\
                            default set is result.jpg')
    args = parser.parse_args()
    return args

def maxfilter(img, kernel):
    h, w = img.shape
    filtered_array = -np.ones((img.shape))
    kernel_size = kernel
    assert kernel_size % 2 != 0
    padding = kernel // 2
    a = np.pad(img, ((padding, ), (padding, )), 'constant', constant_values=(-1, -1))
    h_s, w_s = 0 + padding, 0 + padding
    for i in range(h):
        for j in range(w):
            center = (h_s + i, w_s + j)
            if a[center] != 300:
                max_ele = np.amax(a[center[0] - padding: center[0] + padding + 1, center[1] - padding: center[1] + padding + 1])
                filtered_array[center[0] - padding, center[1] - padding] = max_ele
    return filtered_array

def minfilter(img, kernel):
    h, w = img.shape
    filtered_array = np.ones((img.shape)) * 300
    kernel_size = kernel
    assert kernel_size % 2 != 0
    padding = kernel // 2
    a = np.pad(img, ((padding, ), (padding, )), 'constant', constant_values=(300, 300))
    h_s, w_s = 0 + padding, 0 + padding
    for i in range(h):
        for j in range(w):
            center = (h_s + i, w_s + j)
            if a[center] != -1:
                min_ele = np.amin(a[center[0] - padding: center[0] + padding + 1, center[1] - padding: center[1] + padding + 1])
                filtered_array[center[0] - padding, center[1] - padding] = min_ele
    return filtered_array
def get_result(background, origin):
    enchanced = origin - background
    enchanced = cv2.normalize(enchanced,None, 0, 255, norm_type=cv2.NORM_MINMAX)
    return enchanced
def show_picture(origin, enchanced):
    import matplotlib.pyplot as plt
    plt.subplot(1,2,1)
    plt.imshow(origin, cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(enchanced, cmap = 'gray')
    plt.show()

def run(image, filter_size1, filter_size2, mode):
    if mode == 0:
        A = maxfilter(img, filter_size1)
        B = minfilter(A, filter_size2)
    else:
        A = minfilter(img, filter_size1)
        B = maxfilter(A, filter_size2)

    result = get_result(B, image)
    return result

if __name__ == "__main__":
    args = parse_args()
    filename = args.filename
    output = args.dst
    mode = args.mode
    filter_size1 = args.size1
    filter_size2 = filter_size1 if args.size2 is None else args.size2
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    import time
    t0 = time.time()
    result = run(img,filter_size1, filter_size2, mode)
    print('takes {} s'.format(time.time() - t0))
    cv2.imwrite('images/middle.jpg', img)
    cv2.imwrite(output, result)
    if args.show_flag:
        show_picture(img, result)
