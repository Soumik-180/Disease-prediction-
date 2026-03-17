import pandas as pd

df = pd.read_csv("Data/Cleaned/ckd_cleaned_dataset_v3.csv")

print(df["outcome"].value_counts())
print(df["outcome"].value_counts(normalize=True))