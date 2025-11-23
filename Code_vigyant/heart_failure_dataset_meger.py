import pandas as pd

# -----------------------------
# 1. Load the two CSV files
# -----------------------------
df1 = pd.read_csv("/Users/vigyantnayak/heart_disease_classification/Dataset/heart_failure/heart_failure_clinical_records_dataset.csv")     # contains DEATH_EVENT
df2 = pd.read_csv("/Users/vigyantnayak/heart_disease_classification/Dataset/heart_failure/heart.csv")    # contains HeartDisease

# -----------------------------
# 2. Standardize column names
# -----------------------------
df1.columns = df1.columns.str.lower()
df2.columns = df2.columns.str.lower()

# -----------------------------
# 3. Make sure target columns match
# -----------------------------
# df1 → death_event
# df2 → heartdisease → rename to death_event
if "heartdisease" in df2.columns:
    df2 = df2.rename(columns={"heartdisease": "death_event"})

# -----------------------------
# 4. Combine datasets WITHOUT dropping columns
# -----------------------------
combined_df = pd.concat([df1, df2], axis=0, ignore_index=True, sort=False)

# -----------------------------
# 5. Clean and unify the 'death_event' column
# -----------------------------
# Convert to int, replace missing values with 0
combined_df["death_event"] = combined_df["death_event"].fillna(0).astype(int)

# -----------------------------
# 6. Add a sequential patient ID column
# -----------------------------
combined_df.insert(0, "patient_id", range(1, len(combined_df) + 1))

# -----------------------------
# 7. Save the final dataset
# -----------------------------
combined_df.to_csv("/Users/vigyantnayak/heart_disease_classification/Dataset/heart_failure/combined_heart_failure_final.csv", index=False)

print("Combined dataset saved as combined_heart_failure_final.csv")