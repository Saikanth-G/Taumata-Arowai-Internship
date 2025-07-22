import pandas as pd

# Load the dataset 
df = pd.read_excel("final_combined_water_data.xlsx", sheet_name="Sheet1")

# Step 1: Drop empty or unnamed columns
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

# Step 2: Drop columns with more than 90% missing values
threshold = 0.9
missing_fraction = df.isnull().mean()
df = df.loc[:, missing_fraction < threshold]

# Step 3: Drop duplicate rows
df.drop_duplicates(inplace=True)

# Step 4: Convert date/time-like columns to datetime
for col in df.columns:
    if "date" in col.lower() or "time" in col.lower():
        df[col] = pd.to_datetime(df[col], errors="coerce")

# Step 5: Attempt to clean numeric columns stored as objects
for col in df.select_dtypes(include="object").columns:
    try:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    except:
        pass

# Step 6: Reset index
df.reset_index(drop=True, inplace=True)

# Save the cleaned dataset to Excel
cleaned_file_path = "cleaned_combined_water_data.xlsx"
df.to_excel(cleaned_file_path, index=False)
print(f"✅ Cleaned dataset saved to: {cleaned_file_path}")

