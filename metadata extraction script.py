import pandas as pd
import os

def inspect_file(file_path):
    ext = os.path.splitext(file_path)[-1].lower()

    if ext == '.csv':
        print(f"\n📄 File: {file_path} (CSV)")
        df = pd.read_csv(file_path)
        summarize_sheet(df)
        
    elif ext in ['.xls', '.xlsx']:
        print(f"\n📄 File: {file_path} (Excel)")
        xls = pd.ExcelFile(file_path)
        print(f"📑 Sheets found: {xls.sheet_names}")
        
        for sheet in xls.sheet_names:
            print(f"\n🔍 Inspecting sheet: {sheet}")
            df = xls.parse(sheet)
            summarize_sheet(df)
    else:
        print(f"❌ Unsupported file type: {ext}")

def summarize_sheet(df):
    if df.empty:
        print("⚠️ This sheet is empty.")
        return

    print("\n🧱 Column Names:")
    print(list(df.columns))

    print("\n🔢 Data Types:")
    print(df.dtypes.to_dict())

    print("\n🧪 First Row Data (as string):")
    first_row = df.iloc[0].astype(str).to_dict()
    for col, val in first_row.items():
        print(f"{col}: {val}")

# Example usage
file_path = "River water quality state and trend results (Sept 2024).xlsx"  # Replace with your Excel or CSV file
inspect_file(file_path)
