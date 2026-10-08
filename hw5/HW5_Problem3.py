import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

format_opt = "{:.4f}".format
np.set_printoptions(formatter={'float_kind':format_opt})

p = np.array([
    [0, 1/2, 1/2, 0, 0, 0, 0, 0],
    [0, 0, 1/2, 1/2, 0, 0, 0, 0],
    [1/2, 0, 0, 0, 1/2, 0, 0, 0],
    [1/3, 0, 1/3, 0, 1/3, 0, 0, 0],
    [0, 1/3, 0, 0, 0, 1/3, 1/3, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 0]
])

q = 1/8*np.array([1,1,1,1,1,1,1,1])

r1 = q @ np.linalg.matrix_power(p,100)
r2 = r1 @ p

print(f"q_100: {r1}")
print(f"q_101: {r2}")

vec_1 = np.ones(8)
G = lambda d: (1-d)/8*np.outer(vec_1, vec_1) + d*p
Gd = G(0.85)
q_0 = q
q_n = q_0 @ Gd
i = 0
while np.linalg.norm(q_n - q_0) > 10**(-10):
    q_0 = q_n 
    q_n = q_0 @ Gd
    i += 1
print(f"est. pi: {q_n}")
print(f"iterations: {i}")

A = Gd - np.eye(8)
A = np.c_[A, np.ones(8)].T
b = np.zeros(9)
b[-1] = 1
res = np.linalg.lstsq(A,b)
print(f"true pi:{res[0]}")

def chain_simulation(state, P, N=10**2, T=10**5):
    states = np.full(N, state, dtype=int)
    states_time = np.empty((T,N))
    frequencies = np.empty((T,8))
    for t in range(T):
        u = np.random.rand(N)

        next_states = np.sum(
            u[:, None] > P[states],
            axis=1
        )

        states = next_states
        states_time[t] = states
        frequencies[t] = np.bincount(
            states, minlength=8
        )/N

    return states, states_time, frequencies

FGd = np.cumsum(Gd, axis=1)
N = 10**2
T = 10**5
distribution, state_records, frequencies = chain_simulation(1, FGd, N=N, T=T)

fig, axs = plt.subplots(1,2)
axs[0].hist(state_records[:,0], density=True)
axs[1].hist([0,1,2,3,4,5,6,7], density=True, weights=res)
axs[0].set_title("Estimated Stationary Distribution")
axs[1].set_title("True Stationary Distribution")
fig.supxlabel("States")
fig.supylabel("Relative Frequency of States")
fig.suptitle("Estimated vs True Stationary Distirbution")

X = np.ones(T).cumsum()
Y = np.array([np.linalg.norm(x - res, ord=1) for x in frequencies])
p, logC = np.polyfit(
    np.log(X[10**2:10**5]),
    np.log(Y[10**2:10**5]),
    1
)
f = lambda x: p*x + logC
fX = f(X[10**2:10**5])
plt.loglog(X,Y, label=r'$\mathrm{max}_i|\hat \pi_i - \pi_i|$')
plt.loglog(X, res[5]*(1 - res[5])/X, color='red', label=r'$\sqrt{\pi_6(1 - \pi_6)/T}$')
plt.plot(X[10**2:10**5], fX, color='orange', label=f"{r'$p='}{p:.4f}{r'$'}")
plt.xlabel('log(time)')
plt.ylabel('log(error)')
plt.title('log-log plot of error vs time')
plt.legend()
