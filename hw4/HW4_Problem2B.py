import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

X = np.arange(1, 10**4/2 + 1)
f = lambda x: 1/np.sqrt(np.pi*x)
fX = f(X)
print(f"Expected visits in 10^4 steps: {fX.sum()}")
