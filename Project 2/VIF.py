import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

df = pd.read_csv('project2.csv')

X = df.drop(columns=["price"])

X = pd.get_dummies(X, drop_first=True, dtype=int)

vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns

vif_data["VIF"] = [
    variance_inflation_factor(X.values, i)
    for i in range(X.shape[1])
]

print("Variance Inflation Factor (VIF) Results:")
print(vif_data.sort_values("VIF", ascending=False).to_string(index=False))