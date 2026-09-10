import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon, uniform, zscore

N = 10**5
U = uniform()
f1 = expon(scale=1/1000)
f2 = expon(scale=1/10)

f = lambda t: 0.9*1000*f1.pdf(t) + (1-0.9)*f2.pdf(t)
X = np.linspace(0,20,1000)
Y = f(X)

def composite_simulation():
    u1 = U.rvs()
    if u1 < 0.9:
        u2 = U.rvs()
        x = f1.ppf(u2)
    else:
        u2 = U.rvs()
        x = f2.ppf(u2)
    return x

times = np.array([composite_simulation() for _ in range(N)])

mean = times.mean()
empirical = (times > 0.05).sum()/len(times)

plt.hist(times)
plt.plot(X,Y, label=r'$f(x)$')
plt.xlabel("p")
plt.ylabel("log-frequency")
plt.yscale("log")
plt.title(f"Composite, {r'$\hat\mu=$'}{mean:3f}, {r'$P(T > 50)=$'}{empirical}")
plt.legend()
