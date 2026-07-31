import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson


df = pd.read_csv('project2.csv')

X_all = pd.get_dummies(df.drop(columns=["price"]), drop_first=True, dtype=int)
y = df["price"]

selected_features = [
    'house_size', 
    'number_of_rooms', 
    'distance_to_the_beach', 
    'school_district'
]

X_final = X_all[selected_features]
X_final = sm.add_constant(X_final)

final_model = sm.OLS(y, X_final).fit()

bp_test = het_breuschpagan(
    final_model.resid,
    final_model.model.exog
)

bp_labels = [
    "LM Statistic",
    "LM p-value",
    "F Statistic",
    "F p-value"
]

print("\nBreusch-Pagan Test Results:")
for label, value in zip(bp_labels, bp_test):
    print(f"{label}: {value:.6f}")

dw_stat = durbin_watson(final_model.resid)

print("\nDurbin-Watson Test Result:")
print(f"Durbin-Watson Statistic: {dw_stat:.6f}")