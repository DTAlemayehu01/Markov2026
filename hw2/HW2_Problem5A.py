import numpy as np
x = np.random.random(10**5)
H = lambda z: 1 - np.sqrt(1 - z)
u = H(x).mean()
print(u)
