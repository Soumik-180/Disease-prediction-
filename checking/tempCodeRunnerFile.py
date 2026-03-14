
# for col in df.columns:
#     if col != "outcome":
#         if df.groupby(col)["outcome"].nunique().max() == 1:
#             print("Potential leakage:", col)