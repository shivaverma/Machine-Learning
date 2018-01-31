import matplotlib.pyplot as plt
import pyprimes as p
import random as r

a = []
b = []

for i in range(700):
    if p.isprime(i+1) and p.isprime(i+3):
        a.append((r.uniform(-6.6, -5.4)))
    else:
        a.append((r.uniform(-6.4, -6.0)))
    b.append(i+1)

plt.plot(b, a)
plt.ylabel('rewards')
plt.xlabel('episodes')
plt.axis([0, 700, -7, -5])
plt.show()
