import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('project2.csv')

X = pd.get_dummies(df.drop(columns=["price"]), drop_first=True, dtype=int)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
residuals = y_test - predictions

plt.figure(figsize=(8, 6))

stats.probplot(
    residuals,
    dist="norm",
    plot=plt
)

plt.title("QQ Plot")
plt.tight_layout()
plt.show()