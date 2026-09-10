import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import expon, uniform, zscore
N = 10**4

f = lambda x: x*np.e**(-x)
X = np.linspace(0,10, 1000)
Y = f(X)
U = uniform()

def acceptance_regime(f,g,c):
    t0 = time.time()
    reject = True
    while reject:
        y = g.rvs()
        u = U.rvs()
        ratio = f(y)/c/g.pdf(y)
        if u <= ratio:
            tf = time.time()
            times.append(tf-t0)
            return y
times = []
lmbda = 1/2
c1 = 1/(lmbda*(1 - lmbda)*np.e)
g = expon(lmbda)
x1 = np.array([acceptance_regime(f, g, c1) for _ in range(N)])
Ext1 = np.array(times).mean()

times = []
lmbda = 0.2
c2 = 1/(lmbda*(1 - lmbda)*np.e)
g = expon(lmbda)
x2 = np.array([acceptance_regime(f, g, c2) for _ in range(N)])
Ext2 = np.array(times).mean()

plt.hist(x1, bins=40, density=True)
plt.plot(X,Y, label=r'$f(x)$')
plt.xlabel("p")
plt.ylabel("frequency")
plt.title(f"Acceptance-rejection regime, c = {1/c1:.4f}, mean time per sample = {Ext1:.4f}")
plt.legend()

plt.hist(x2, bins=40, density=True)
plt.plot(X,Y, label=r'$f(x)$')
plt.xlabel("p")
plt.ylabel("frequency")
plt.title(f"Acceptance-rejection regime, c = {1/c2:.4f}, mean time per sample = {Ext2:.4f}")
plt.legend()
