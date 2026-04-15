import pandas as pd
import os

file_path = r"Q:\2025\2025_03_13_NDL_drone recordings\1-Info\NDL20250311.xlsx"
if os.path.exists(file_path):
    try:
        xl = pd.ExcelFile(file_path)
        print("Sheet Names:", xl.sheet_names)
        for sheet in xl.sheet_names:
            print(f"\n--- Sheet: {sheet} ---")
            df = pd.read_excel(file_path, sheet_name=sheet)
            print(df.head(10))
    except Exception as e:
        print(f"Error: {e}")
else:
    print(f"File not found: {file_path}")
