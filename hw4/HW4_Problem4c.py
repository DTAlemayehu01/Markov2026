import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict 

# reduced into (U, I, M, {N, N'}, A)
p = np.array([
    [0, 1/2, 1/2, 0, 0],
    [1/4, 0, 0, 1/2, 1/4],
    [3/4, 0, 0, 1/2, 1/4],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1],
])
P = np.cumsum(p, axis=1)

def absorbed(x):
    mask = np.array([1,1,1,0,0,0])
    return (x & mask).all()

def chain_simulation(state, N=10**4):
    states = np.full(N, state, dtype=int)
    times = np.zeros(N, dtype=int)
    alive = np.ones(N, dtype=bool)
    
    while alive.any():
        u = np.random.rand(alive.sum())
        curr_state = states[alive]

        next_states = np.sum(
            u[:, None] > P[curr_state],
            axis=1
        )

        states[alive] = next_states
        times[alive] += 1

        alive &= states < 3

    return times, states

N=10**4
U_res = chain_simulation(0, N=N)
I_res = chain_simulation(1, N=N)
M_res = chain_simulation(2, N=N)

hathu = (U_res[1] == 3).sum()/N
hathi = (I_res[1] == 3).sum()/N
hathm = (M_res[1] == 3).sum()/N

hatgu = U_res[0].sum()/N
hatgi = I_res[0].sum()/N
hatgm = M_res[0].sum()/N

tau = lambda x, y : (x[0] * (x[1] == y)).sum()/(x[1] == y).sum()
hatfu = tau(U_res, 3)
hatfi = tau(I_res, 3)
hatfm = tau(M_res, 3)

hatau = tau(U_res, 4)
hatai = tau(I_res, 4)
hatam = tau(M_res, 4)

df = pd.DataFrame({
    'f':['h_x','g_x','t_x^F', 't_x^A', 'hat h_x','hat g_x','hat t_x^F', 'hat t_x^A'],
    'U':[1/2, 4, 4, 4, hathu, hatgu, hatfu, hatau],
    'I':[5/8, 2, 9/5, 7/3, hathi, hatgi, hatfi, hatai],
    'M':[3/8, 4, 5, 17/5, hathm, hatgm, hatfm, hatam],
})
df

# Further reduced to (U, I, M, {F,A})
Q = p[0:3,:3]
R = p[0:3,3:]
X = np.arange(1,16,1, dtype=int)
f = lambda x: np.linalg.matrix_power(Q,x-1) @ R
plt.hist(I_res[0][I_res[1] == 3], density=True)
plt.plot(X,[f(x)[1][0] for x in X], marker='o')
plt.xlabel("Time steps")
plt.ylabel("P(T = n)")
plt.title("PMF for Folding From State I")

# Further reduced to (U, I, M, {F,A})
plt.hist(I_res[0][I_res[1] == 4], density=True)
plt.plot(X,[f(x)[1][1] for x in X], marker='o')
plt.xlabel("Time steps n")
plt.ylabel("P(T = n)")
plt.title("PMF for Aggregating From State I")
