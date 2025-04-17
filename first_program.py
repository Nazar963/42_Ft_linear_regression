import json
import pandas as pd

with open("model_parameters.json", "r") as f:
    params = json.load(f)
theta0 = params["theta0"]
theta1 = params["theta1"]
with open("data.csv", "r") as f:
    data = pd.read_csv(f)
    X_mean = data["km"].mean()
    X_std = data["km"].std()
    y_mean = data["price"].mean()
    y_std = data["price"].std()

while True:
    km = input("Enter km: ")
    if not km.isnumeric() or float(km) < 0:
        print("Invalid input. Please enter a positive number.")
        continue
    km = float(km)
    km_norm = (km - X_mean) / X_std
    price_norm = theta0 + theta1 * km_norm
    price = price_norm * y_std + y_mean
    print(f"Prediction for km={km}: {price}")
    another = input("Do you want to enter another km? (yes/no): ")
    if another.lower() != 'yes':
        break