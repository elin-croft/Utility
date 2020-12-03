from im2col_cython import im2col
import numpy as np

img = np.ones((2,3))
padding = 2
a = np.pad(img, ((padding, ), (padding, )), 'constant', constant_values=(300, 300))
output = im2col(a,1, 2, 3, 3)
print(output.shape)