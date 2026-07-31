import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv("project2.csv")

X = df[[
    "house_size",
    "size_of_the_yard",
    "distance_to_the_beach"
]]

y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

r2 = r2_score(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

print("R² =", round(r2, 4))
print("RMSE =", round(rmse, 2))