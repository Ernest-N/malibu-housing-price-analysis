import pandas as pd
import numpy as np
import statsmodels.api as sm

def forward_selection(data, target):
    remaining = list(data.columns)
    selected = []
    current_aic = np.inf

    while remaining:
        aic_candidates = []

        for candidate in remaining:
            predictors = selected + [candidate]
            X_subset = sm.add_constant(data[predictors])
            model = sm.OLS(target, X_subset).fit()

            aic_candidates.append(
                (model.aic, candidate)
            )

        aic_candidates.sort()
        best_aic, best_feature = aic_candidates[0]

        if best_aic < current_aic:
            remaining.remove(best_feature)
            selected.append(best_feature)
            current_aic = best_aic

            print(
                f"Added {best_feature} | AIC = {best_aic:.2f}"
            )
        else:
            break

    return selected

df = pd.read_csv('project2.csv')

X = pd.get_dummies(
    df.drop(columns=["price"]),
    drop_first=True,
    dtype=int             
)
y = df["price"]

selected_features = forward_selection(X, y)

print("\nSelected Features:")
print(selected_features)