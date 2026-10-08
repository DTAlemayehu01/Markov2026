import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

format_opt = "{:.4f}".format
np.set_printoptions(formatter={'float_kind':format_opt})

step_up = lambda a, k: a*(4-k)/4
step_down = lambda b, k: b*k/4
remain = lambda a, b, k: 1 - step_up(a,k) - step_down(b,k)
p = lambda a, b: np.array([
    [remain(a,b,0), step_up(a,0), 0, 0, 0],
    [step_down(b,1), remain(a,b,1), step_up(a,1), 0, 0],
    [0, step_down(b,2), remain(a,b,2), step_up(a, 2), 0],
    [0, 0, step_down(b,3), remain(a,b,3), step_up(a,3)],
    [0, 0, 0, step_down(b,4), remain(a, b,4)],
], dtype=np.double)

def power_iteration(q_0, p, N=60):
    distributions = np.empty((N+1, len(q_0)))
    distributions[0] = q_0
    for t in range(1, N+1):
        distributions[t] = distributions[t-1] @ p

    return distributions
    
P = p(1,1)
q_0 = np.array([1,0,0,0,0])
distributions = power_iteration(q_0, P)
print(f"q_50: {distributions[50]}")
print(f"q_51: {distributions[51]}")

X = np.array(range(61))
Y2 = distributions[:,2]
Y4 = distributions[:,4]
AVG2 = Y2.cumsum()/(X + 1)
plt.plot(X,Y2, color='red', label=r'$q_n(2)$')
plt.plot(X,Y4, color='blue', label=r'$q_n(4)$')
plt.plot(X,AVG2, color='orange', label=r'$\frac{\sum_{k<= n} q_k(2)}{n+1}$') 
plt.xlabel('N')
plt.ylabel('Probability of state')
plt.title('State Probability as a Function of Time')
plt.legend()

P_p = p(1/10,3/10)

A = P_p - np.identity(5)
A = np.c_[A, np.ones(5)].T
b = np.array([0,0,0,0,0, 1])
sol = np.linalg.lstsq(A,b)[0]
print(f"pi: {sol}")


q_0 = np.array([1,0,0,0,0])
q_n = q_0 @ P_p
n = 1
while np.linalg.norm(q_n - sol, ord=1) > 10**(-6):
#for _ in range(400):
    q_n = q_n @ P_p
    n += 1

print(f"Convergence at step {n}")

eigenvalues = np.sort(abs(np.linalg.eigvals(P_p)))
print(f"Second Largest Eigenvalue: {eigenvalues[1]:.3f}")
