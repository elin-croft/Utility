import numpy as np

def im2col(data, channels, height, width,
            kernel):

    pading = kernel // 2
    length = height * width
    if channels == 1:
        output = np.ones((length, kernel ** 2))
    elif channels > 1:
        output = np.ones((channels, length, kernel ** 2))

    for i in range(length):
        h, w = i //width + pading, i % width + pading
        if channels == 1:
            output[i, :] = data[h - pading: h + pading + 1, w - pading: w + pading + 1].ravel()
        elif channels > 1:
            output[0, i, :] = data[h - pading: h + pading + 1, w - pading: w + pading + 1].ravel()
    return output
