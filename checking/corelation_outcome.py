import pandas as pd

df = pd.read_csv("Data/Cleaned/ckd_cleaned_dataset_v3.csv")

corr = df.corr(numeric_only=True)["outcome"].sort_values(ascending=False)

print(corr)

for col in df.columns:
    if col != "outcome":
        if df.groupby(col)["outcome"].nunique().max() == 1:
            print("Potential leakage:", col)