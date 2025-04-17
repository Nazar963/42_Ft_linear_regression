import numpy as np
import json
import pandas as pd

file = pd.read_csv("data.csv")
X = file["km"].values
y = file["price"].values
X = (X - np.mean(X)) / np.std(X)
y = (y - np.mean(y)) / np.std(y)
theta0 = 0          #? Intercept
theta1 = 0          #? Slope
learn_rate = 0.001
iter = 1000
m = len(y)

def compute_cost(predictions, y, m):
    return (1 / (2 * m)) * np.sum((predictions - y) ** 2)

for i in range(iter):
    pred = theta0 + theta1 * X
    theta0 -= (learn_rate / m) * np.sum(pred - y)
    theta1 -= (learn_rate / m) * np.sum((pred - y) * X)
    if i % 100 == 0:
        cost = compute_cost(pred, y, m)
        print(f"Iteration {i}: Cost = {cost}")

params = {
    "theta0": theta0 * np.std(y) + np.mean(y),
    "theta1": theta1 * (np.std(y) / np.std(X))
}
with open("model_parameters.json", "w") as f:
    json.dump(params, f)

print(f"Optimized parameters: theta0 = {params['theta0']}, theta1 = {params['theta1']}")