import numpy as np
import matplotlib.pyplot as plt

eq1 = lambda x, y, z: x**2 + y**2 < z
eq2 = lambda x, y, z: z**2 > x*y
def monte_carlo(trials):
    successes = 0
    for _ in range(trials):
        x, y, z = np.random.rand(3)
        successes = successes + (eq1(x,y,z) and eq2(x,y,z))
    p_hat = successes/trials
    return p_hat

N = list(range(1, 1000))
p1 = [monte_carlo(n) for n in N]

plt.semilogx(N, p1)
