import numpy as np

format_opt = "{:.4f}".format
np.set_printoptions(formatter={'float_kind':format_opt})

p = np.array([
    [0.5, 0.5, 0, 0, 0, 0],
    [1/3, 0, 1/3, 0, 1/3, 0],
    [0, 0, 1/4, 3/4, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 0]
])

print(np.linalg.matrix_power(p,20))
print(np.linalg.matrix_power(p,21))
