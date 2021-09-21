import hnumpy as hnp
import numpy as np

def relu(x):
    return np.maximum(x, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

h_relu = hnp.compile_fhe(
    relu,
    {"x": hnp.encrypted_ndarray(bounds=(0,1), shape=(2,2))}
)

h_sigmoid = hnp.compile_fhe(
    sigmoid,
    {"x": hnp.encrypted_ndarray(bounds=(0,1), shape=(2,2))}
)

x = np.random.uniform(0, 1, (2,2))

res_relu = h_relu.simulate(x)
res_sigmoid = h_sigmoid.simulate(x)

print(f"Homomorphic result: {res_relu}, {res_sigmoid}")
print(f"Real result: {relu(x)}, {sigmoid(x)}")