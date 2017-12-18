import numpy as np

a = np.array([[1, 2, 4, 5],
              [4, 2, 6, 8],
              [2, 9, 0, 3]])

print "rank : " + str(a.ndim)
print "dimensions: " + str(a.shape)
print "size: " + str(a.size) + " elements"
print "data type: " + str(a.dtype)
print "element size: " + str(a.itemsize) + " Byte"

lis = a.tolist()
lis.pop(1)

a = np.reshape(lis, (2, 4))
print "new array: " + str(a)
