import numpy as np

a = np.array([[7,8,9],[4,5,6],[1,2,3]])

a_ = np.zeros(9)

for j in range(3):
    for k in range(3):
        p = j + (k-1)*3
        print(p)
        a_[p] = a[j,k]

print(a_)