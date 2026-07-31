import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("project2.csv")

print(df.info())
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

corr_matrix = df.corr(numeric_only=True)

plt.figure(figsize=(14, 10))

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
    annot_kws={"size": 8} 
)

plt.title("Correlation Matrix", fontsize=16)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()

plt.show()

corr_price = corr_matrix["price"].sort_values(ascending=False)

print("\nCorrelation with Price")
print(corr_price)