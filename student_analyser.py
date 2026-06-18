import pandas as pd

# ── RAW MESSY DATA (real world data is always like this) ──
data = {
    "Name": ["Divya", "Priya", "Sneha", "Anjali", "Meera",
             "Divya", "Riya", "Sneha", None, "Kavya"],
    "City": ["Delhi", "Mumbai", "Delhi", "Pune", "Mumbai",
             "Delhi", "Chennai", "Delhi", None, "Pune"],
    "Subject": ["Python", "ML", "Python", "DSA", "ML",
                "Python", "DSA", "Python", None, "ML"],
    "Marks": [88, 92, 78, 85, None,
              88, 90, 78, 70, 95],
    "Attendance": [90, 85, 92, 78, 88,
                   90, 95, 92, 60, 88]
}

df = pd.DataFrame(data)

print("=" * 50)
print(" RAW DATA — Before Cleaning")
print("=" * 50)
print(df)

# ── 1. ISNULL — find missing values ──
print("\n" + "=" * 50)
print("STEP 1 — Finding Missing Values (isnull)")
print("=" * 50)
print(df.isnull().sum())

# ── 2. NOTNULL — show only complete rows ──
print("\n" + "=" * 50)
print(" STEP 2 — Complete Rows Only (notnull)")
print("=" * 50)
complete = df[df["Marks"].notnull() & df["Name"].notnull()]
print(complete)

# ── 3. DROPNA — drop all rows with any missing value ──
print("\n" + "=" * 50)
print(" STEP 3 — Drop Missing Values (dropna)")
print("=" * 50)
df_clean = df.dropna()
print(df_clean)
print(f"Rows before: {len(df)} → Rows after: {len(df_clean)}")

# ── 4. DROP DUPLICATES ──
print("\n" + "=" * 50)
print("STEP 4 — Remove Duplicates (drop_duplicates)")
print("=" * 50)
df_clean = df_clean.drop_duplicates()
print(df_clean)
print(f"Rows after removing duplicates: {len(df_clean)}")

# ── 5. UNIQUE & NUNIQUE ──
print("\n" + "=" * 50)
print(" STEP 5 — Unique Values (unique & nunique)")
print("=" * 50)
print(f"Unique cities: {df_clean['City'].unique()}")
print(f"Number of unique cities: {df_clean['City'].nunique()}")
print(f"Unique subjects: {df_clean['Subject'].unique()}")
print(f"Number of unique subjects: {df_clean['Subject'].nunique()}")

# ── 6. VALUE_COUNTS ──
print("\n" + "=" * 50)
print(" STEP 6 — Subject Frequency (value_counts)")
print("=" * 50)
print(df_clean["Subject"].value_counts())
print("\nCity wise students:")
print(df_clean["City"].value_counts())

# ── 7. SORT_VALUES ──
print("\n" + "=" * 50)
print(" STEP 7 — Sorted by Marks (sort_values)")
print("=" * 50)
sorted_df = df_clean.sort_values("Marks", ascending=False)
print(sorted_df[["Name", "Marks", "Attendance"]])

# ── 8. RANK ──
print("\n" + "=" * 50)
print(" STEP 8 — Student Rankings (rank)")
print("=" * 50)
df_clean = df_clean.copy()
df_clean["Rank"] = df_clean["Marks"].rank(ascending=False).astype(int)
print(df_clean[["Name", "Marks", "Rank"]].sort_values("Rank"))

# ── 9. SET_INDEX ──
print("\n" + "=" * 50)
print(" STEP 9 — Set Name as Index (set_index)")
print("=" * 50)
df_indexed = df_clean.set_index("Name")
print(df_indexed)

# ── 10. SORT_INDEX ──
print("\n" + "=" * 50)
print(" STEP 10 — Sort by Index/Name (sort_index)")
print("=" * 50)
print(df_indexed.sort_index())

# ── 11. RESET_INDEX ──
print("\n" + "=" * 50)
print(" STEP 11 — Reset Index back (reset_index)")
print("=" * 50)
df_final = df_indexed.reset_index()
print(df_final)

# ── FINAL SUMMARY ──
print("\n" + "=" * 50)
print(" FINAL ANALYSIS SUMMARY")
print("=" * 50)
print(f"Total students: {len(df_final)}")
print(f"Average marks: {df_final['Marks'].mean():.2f}")
print(f"Highest marks: {df_final['Marks'].max()}")
print(f"Topper: {df_final.loc[df_final['Marks'].idxmax(), 'Name']}")
print(f"Average attendance: {df_final['Attendance'].mean():.2f}%")
print("\nSubject wise average marks:")
print(df_final.groupby("Subject")["Marks"].mean().sort_values(ascending=False))