import numpy as np
cimport numpy as np
cimport cython

def im2col(np.ndarray[unsigned char, ndim=2] data, 
                int channels, int height, int width, int kernel):
    cdef np.ndarray[unsigned char,ndim=2] output = im2col_core(data, channels, height, width, kernel)

    return output

cdef np.ndarray[unsigned char,ndim=2] im2col_core(np.ndarray[unsigned char, ndim=2] data, 
                int channels, int height, int width, int kernel):
    cdef int length = height * width
    cdef np.ndarray[unsigned char, ndim=2] output
    cdef int offset_h
    cdef int offset_w
    cdef int HH
    cdef int WW

    if channels == 1:
        output = np.ones((length, kernel * kernel), dtype=np.uint8)
    elif channels == 3:
        output = np.ones((channels, length, kernel * kernel), dtype=np.uint8)

    for i in range(length):
        offset_h, offset_w = i // width, i % width
        for j in range(kernel * kernel):
            HH, WW = j // kernel, j % kernel
            output[HH, WW] = data[HH + offset_h, WW + offset_w]
    return output