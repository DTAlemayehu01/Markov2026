import numpy as np
f = lambda s, a, b: a/2 + (1-a)*1/(1 + np.exp(-b*s))
f_model = lambda s: f(s, 0.2, 1)
print(f_model(2))

numerator = 0.2/2
denom = (1 - f_model(2))
print(numerator/denom)
