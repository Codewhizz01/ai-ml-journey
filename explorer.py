import pandas as pd
#-----series demo----------
print("="*40)
print("series basics")
print("="*40)
scores=pd.Series([85,92,78,95,88],index=["maths","science","english","python","dsa"])
print(scores)
print(f"\nHighest score:{scores.max()}")
print(f"subjects with highest score:{scores.idxmax()}")
print(f"average score:{scores.mean():.2f}")
print(f"subjects above 85:\n{scores[scores>85]}")

# ------DATAFRAME DEMO -----------
print("\n" + "=" * 40)
print("DATAFRAME BASICS")
print("=" * 40)

data = {
    "Name":    ["Riya", "Priya", "Sneha", "Anjali", "Meera"],
    "Age":     [21, 22, 20, 23, 21],
    "City":    ["Delhi", "Mumbai", "Delhi", "Pune", "Mumbai"],
    "CGPA":    [8.5, 9.1, 7.8, 8.9, 9.4],
    "Placed":  [True, True, False, True, False]
}

df = pd.DataFrame(data)

print("\n--- First Look ---")
print(df.head())

print("\n--- Shape ---")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n--- Data Types ---")
print(df.dtypes)

print("\n--- Statistics ---")
print(df.describe())

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Filter: CGPA above 8.5 ---")
print(df[df["CGPA"] > 8.5])

print("\n--- Group by City: Average CGPA ---")
print(df.groupby("City")["CGPA"].mean())

print("\n--- Sorted by CGPA ---")
print(df.sort_values("CGPA", ascending=False))

print("\n--- New Column: Grade ---")
df["Grade"] = df["CGPA"].apply(
    lambda x: "Distinction" if x >= 9 else "First Class"
)
print(df[["Name", "CGPA", "Grade"]])

print("\n--- Placed Students Only ---")
print(df[df["Placed"] == True][["Name", "City", "CGPA"]])