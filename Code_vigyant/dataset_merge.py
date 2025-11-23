import pandas as pd

df = pd.read_csv("/Users/vigyantnayak/heart_disease_classification/Dataset/heart_failure/combined_heart_failure_final.csv")

# Fix column name issues: lowercase + strip spaces
df.columns = df.columns.str.lower().str.strip()

# Convert all values in the sex column to strings and strip spaces
df["sex"] = df["sex"].astype(str).str.strip()

# Replace "0" → "F", "1" → "M"
df["sex"] = df["sex"].replace({"0": "F", "1": "M"})

df.to_csv("/Users/vigyantnayak/heart_disease_classification/Dataset/heart_failure/combined_heart_failure_final.csv", index=False)

print("Sex column updated successfully ✔️")
