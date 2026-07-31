import pandas as pd
import statsmodels.api as sm

df = pd.read_csv("project2.csv")

X = df[[
    "house_size",
    "size_of_the_yard",
    "distance_to_the_beach"
]]

y = df["price"]

X = sm.add_constant(X)

final_model = sm.OLS(y, X).fit()

# R² and Adjusted R²
print("R²:", final_model.rsquared)
print("Adjusted R²:", final_model.rsquared_adj)

# F Statistic
print("F-statistic:", final_model.fvalue)

# p-values
print("\nP-values:")
print(final_model.pvalues)

# Coefficients
print("\nCoefficients:")
print(final_model.params)