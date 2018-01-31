import matplotlib.pyplot as plt
import pyprimes as p
import random as r

a = []
b = []

for i in range(1000):

    if i< 150:
        a.append(r.uniform(-15, 5))

    elif i>150 and i<200:
        a.append(r.uniform(-11, 3))
    elif p.isprime(i):
        a.append(r.uniform(-8, -1))
    else:
        a.append((r.uniform(-8, -3)))
    b.append(i+1)

plt.plot(b, a)
plt.ylabel('rewards')
plt.xlabel('episodes')
plt.axis([0, 1000, -15, 5])
plt.show()
