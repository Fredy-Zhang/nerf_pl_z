from numba import prange

import torch
import numpy as np

B = 190512
a = torch.arange(B)

def reange(x, B, index):
    _result = np.zeros(B)
    for idx in prange(B):  ## 0-504*378-1
        #print("idx: ", idx)
        _result[idx] = x[list(index).index(idx)]
    return torch.tensor(_result)

index = np.arange(B)
print(index)
np.random.shuffle(index)
print(torch.tensor(index))
b = a[index]
print(a)
import datetime
# s = datetime.datetime.now()
# print(reange(b, 2000, index))
# e = datetime.datetime.now()
# print(": ", e - s)
#print(torch.gather(b, -1, torch.tensor(index)))

s = datetime.datetime.now()
c = zip(index, b)
print(a)
print(b)
d = sorted(c, key=lambda t:t[0])
print(torch.tensor(list(map(lambda x:x[1].tolist(), d))))
print(torch.tensor(list(map(lambda x:x[1].tolist(), sorted(zip(index, b),key=lambda t:t[0])))))
e = datetime.datetime.now()
print(": ", e - s)