import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan


df = pd.read_csv("project2.csv")

y = np.log(df["price"])

X = pd.get_dummies(
    df.drop(columns=["price"]),
    drop_first=True,
    dtype=int
)

features_to_drop = [
    "lot_size"
]

X = X.drop(
    columns=[c for c in features_to_drop if c in X.columns],
    errors="ignore"
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

X_train_const = sm.add_constant(X_train)
X_test_const = sm.add_constant(X_test)

model = sm.OLS(
    y_train,
    X_train_const
).fit(
    cov_type="HC3"
)

print(model.summary())

dw = durbin_watson(model.resid)

print("\nDurbin-Watson Test")
print(f"Durbin-Watson Statistic: {dw:.4f}")

if 1.5 <= dw <= 2.5:
    print("Interpretation: Little evidence of autocorrelation.")
elif dw < 1.5:
    print("Interpretation: Positive autocorrelation may be present.")
else:
    print("Interpretation: Negative autocorrelation may be present.")

bp_lm, bp_pvalue, bp_f, bp_f_pvalue = het_breuschpagan(model.resid, X_train_const)

print("\nBreusch-Pagan Test (Final Model)")
print(f"LM Statistic: {bp_lm:.4f}")
print(f"LM p-value: {bp_pvalue:.4f}")
print(f"F Statistic: {bp_f:.4f}")
print(f"F p-value: {bp_f_pvalue:.4f}")

if bp_pvalue < 0.05:
    print("Interpretation: Heteroscedasticity is still present in the data, but HC3 robust standard errors ensure valid inference.")
else:
    print("Interpretation: No significant heteroscedasticity detected.")

log_predictions = model.predict(X_test_const)

predictions = np.exp(log_predictions)
actual = np.exp(y_test)

r2 = r2_score(actual, predictions)

rmse = np.sqrt(
    mean_squared_error(
        actual,
        predictions
    )
)

print("\nImproved Model:")
print(f"R²: {r2:.4f}")
print(f"RMSE: ${rmse:,.0f}")

vif_df = pd.DataFrame()

vif_df["Feature"] = X.columns

vif_df["VIF"] = [
    variance_inflation_factor(
        X.values,
        i
    )
    for i in range(X.shape[1])
]

print("\nVIF:")
print(
    vif_df.sort_values(
        by="VIF",
        ascending=False
    )
)