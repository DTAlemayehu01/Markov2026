import numpy as np
import matplotlib.pyplot as plt

eq = lambda x, y, z: x**2 + y**2 < z
inverse_transform = lambda x : np.sqrt(x)

def monte_carlo(f, trials):
    successes = 0
    for _ in range(trials):
        x, y, z = np.random.rand(3)
        x, y = inverse_transform([x, y])
        successes = successes + f(x,y,z)
    p_hat = successes/trials
    return p_hat

N = list(range(1, 1000))
p1 = [monte_carlo(eq, n) for n in N]

plt.semilogx(N, p1)
