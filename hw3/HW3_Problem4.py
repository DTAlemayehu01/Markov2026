import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import erf
from scipy.stats import linregress

def simulation(R, N, T, r=1):
    distance_matrix = np.matrix([[10 for _ in range(R)] for _ in range(N)])
    live_sheep = np.array([True for _ in range(R)])
    steps = [-2, 0, 2]
    p = [0.25, 0.5, 0.25]
    portion_alive = []
    for t in range(T):
        for lion in range(N):
            update = np.random.choice(steps, p=p, size=R)
            distance_matrix[lion] = distance_matrix[lion] + update
        deaths = np.absolute(distance_matrix).min(axis=0)
        live_sheep = np.logical_and(live_sheep, deaths)
        portion_alive.append(live_sheep.sum()/R)
    return np.array(portion_alive)
        
R = 2*10**4
N = 1
T = 10**4
p1 = simulation(R, N, T)
b1 = linregress(
    np.log(np.array(range(1,T+1))[10**2:10**4]),
    np.log(
        (p1 + 0.001)[10**2:10**4]
    )
).slope
print(b1)

N = 2
p2 = simulation(R, N, T)
b2 = linregress(
    np.log(np.array(range(1,T+1))[10**2:10**4]),
    np.log(
        (p2 + 0.001)[10**2:10**4]
    )
).slope
print(b2)

fig, ax = plt.subplots(1,2, figsize=(10,10))

x = np.array(range(1,T+1))
S1 = lambda t: erf(10/2/np.sqrt(t))
f1 = lambda t: t**(b1)
f2 = lambda t: t**(b2)
fig.suptitle(f"Fit Window: {r'$x \in [10^2, 10^4]$'}")
ax[0].loglog(x[10**2:10**4], p1[10**2:10**4], label=r"$\hat S_1(t)$")
ax[0].loglog(x[10**2:10**4], S1(x)[10**2:10**4], label=r"$S_1(t)$")
ax[0].loglog(x[10**2:10**4], f1(x)[10**2:10**4], label=r"$\beta_1$")
ax[0].set_title(f"{r'$\beta_1$'} = {b1:.4f}")
ax[0].set_xlabel(f"{r'$t$'}")
ax[0].set_ylabel(f"{r'$S_1(t)$'}")
ax[1].loglog(x[10**2:10**4], p2[10**2:10**4], label=r"$\hat S_2(t)$")
ax[1].loglog(x[10**2:10**4], (S1(x)**2)[10**2:10**4], label=r"$S_1(t)^2$")
ax[1].loglog(x[10**2:10**4], f2(x)[10**2:10**4], label=r"$\beta_2$")
ax[1].set_title(f"{r'$\beta_2$'} = {b2:.4f}")
ax[0].set_xlabel(f"{r'$t$'}")
ax[0].set_ylabel(f"{r'$S_2(t)$'}")
ax[0].legend()
ax[1].legend()

t = np.array([10**2, 10**3, 10**4])
s1t = f1(t)**2
s2t = f2(t)
df = pd.DataFrame({'t':t, r'S1^2':s1t, r'S2':s2t})
df
