import numpy as np

format_opt = "{:.4f}".format
np.set_printoptions(formatter={'float_kind':format_opt})

p = np.array([
    [0.9, 0.1, 0],
    [0, 0.75, 0.25],
    [0.5, 0, 0.5]
])

print(np.linalg.matrix_power(p,50))
