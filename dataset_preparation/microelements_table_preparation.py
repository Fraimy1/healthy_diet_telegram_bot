"""
This script downloads an Excel file (via a provided Google Drive/Sheets link),
processes a microelements table that contains a header block (the first 7 rows)
followed by product data rows (around 1347+ lines),
and writes out a cleaned CSV file using functional programming principles.

Changes made compared to previous versions:
  1. No unit conversion is performed.
  2. All product data rows (beyond the header block) are retained.
  
Make sure you have installed the packages: pandas and gdown,
and that your DATABASE_PATH is defined in config/config.py.
"""

import os
from functools import partial
import gdown
import pandas as pd
from config.config import DATABASE_PATH

# Define the list of columns that should remain as strings.
non_numeric_columns = [
    'Продукты, назв сокр',
    'БК',
    'Код БК',
    'число продуктов'
]

# ─── STEP 1: DOWNLOAD THE EXCEL FILE ──────────────────────────────────────────
def download_excel(url: str) -> str:
    """
    Downloads an Excel file from a Google Drive/Sheets link using gdown
    and saves it as 'dirty_data.xlsx' in the DATABASE_PATH.
    Returns the local file path.
    """
    dest = os.path.join(DATABASE_PATH, 'dirty_data.xlsx')
    gdown.download(url, dest, quiet=False)
    return dest

# ─── STEP 2: READ THE EXCEL FILE WITHOUT PRE-DEFINED HEADER ────────────────
def load_excel_no_header(path: str) -> pd.DataFrame:
    """
    Reads the Excel file without treating any row as the header.
    """
    return pd.read_excel(path, header=None)

# ─── STEP 3: SPLIT THE HEADER BLOCK FROM THE PRODUCT DATA ───────────────────
def extract_header_and_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Assumes that the first 7 rows (index 0 to 6) contain header metadata:
      Row 0: Short names (e.g. "Продукты, назв сокр", etc.)
      Row 1: Единицы измерения
      Row 2: Коэффициенты к мг
      Row 3: Норма
      Row 4: Полное, но более короткое название
      Row 5: Полное название
      Row 6: Описание
    The rest of the rows contain product data.
    Returns a tuple: (header_block, data_block)
    """
    header_block = df.iloc[:7].reset_index(drop=True)
    data_block = df.iloc[7:].reset_index(drop=True)
    return header_block, data_block

# ─── UTILITY: FIND THE PRODUCT-NAME COLUMN ────────────────────────────────
def get_product_col_index(header_block: pd.DataFrame) -> int:
    """
    Returns the column index where the first row of header_block equals
    "Продукты, назв сокр" (after stripping whitespace). Raises an error if not found.
    """
    for col in range(header_block.shape[1]):
        if str(header_block.iloc[0, col]).strip() == "Продукты, назв сокр":
            return col
    raise ValueError("Product column ('Продукты, назв сокр') not found in header.")

# ─── STEP 4: PROCESS THE PRODUCT DATA ──────────────────────────────────────────
# (Each function takes and returns a DataFrame, allowing functional composition.)

def drop_missing_products(data_block: pd.DataFrame, product_col: int) -> pd.DataFrame:
    """
    Drops rows from the product data that have a missing product name
    in the specified product_col.
    """
    return data_block.dropna(subset=[product_col])

def fill_missing_numeric(data_block: pd.DataFrame, product_col: int, header_names: list) -> pd.DataFrame:
    """
    Fills missing values in all product data columns (except the product name column
    and columns specified in non_numeric_columns) with 0.
    """
    for col in data_block.columns:
        # Skip the product name column and any column that should remain as string.
        if col == product_col or header_names[col] in non_numeric_columns:
            continue
        data_block[col] = data_block[col].fillna(0)
    return data_block

def convert_columns_to_float(data_block: pd.DataFrame, product_col: int, header_names: list) -> pd.DataFrame:
    """
    Converts non-product columns in the product data to float, except for columns
    specified in non_numeric_columns.
    It removes extra commas or spaces from string numbers.
    """
    def convert_val(x):
        try:
            if isinstance(x, str):
                # Remove commas and spaces before conversion.
                return float(x.replace(',', '').replace(' ', ''))
            return float(x)
        except Exception:
            return x

    for col in data_block.columns:
        # Skip columns that should remain as strings.
        if col == product_col or header_names[col] in non_numeric_columns:
            continue
        data_block[col] = data_block[col].apply(convert_val)
    return data_block

# ─── STEP 5: SAVE THE FINAL CSV INCLUDING THE HEADER BLOCK ──────────────────
def save_final_csv(header_block: pd.DataFrame, data_block: pd.DataFrame, output_path: str):
    """
    Concatenates the header block (the 7 metadata rows) and the processed product data,
    and writes the full DataFrame to CSV using '|' as the separator.
    """
    final_df = pd.concat([header_block, data_block], ignore_index=True)
    final_df.columns = final_df.iloc[0]
    final_df.drop(0, inplace=True)
    final_df.to_csv(output_path, index=False, sep='|')

# ─── PIPELINE: COMPOSE THE STEPS FUNCTIONALLY ───────────────────────────────
def process_microelements_table(url: str):
    # Download and read the Excel file.
    file_path = download_excel(url)
    df_raw = load_excel_no_header(file_path)

    # Separate header (rows 0-6) from product data (all remaining rows).
    header_block, data_block = extract_header_and_data(df_raw)

    # Get header names from the first row of header_block.
    header_names = list(header_block.iloc[0])
    # Identify the product name column (by checking the header row).
    product_col = get_product_col_index(header_block)

    # Process the product data:
    # 1. Drop rows missing the product name.
    data_block = drop_missing_products(data_block, product_col)

    # 2. Fill missing numeric values (only for numeric columns).
    data_block = fill_missing_numeric(data_block, product_col, header_names)
    # 3. Convert numeric columns to float (skipping non_numeric_columns).
    data_block = convert_columns_to_float(data_block, product_col, header_names)

    # (No unit conversion or row filtering is performed now.)

    # Save the complete table (header + product data) to CSV.
    output_path = os.path.join(DATABASE_PATH, 'cleaned_microelements_table.csv')
    save_final_csv(header_block, data_block, output_path)

    # For verification, print the first several rows of the saved CSV.
    result = pd.read_csv(output_path, sep='|')
    print(result.head(10))

# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Replace with the actual link to your Excel file.
    excel_url = "https://docs.google.com/spreadsheets/d/1I151JmSv9oJT-wsyDkPoHU4yTVcCAx66WicJYULqNDA/export?format=xlsx"
    process_microelements_table(excel_url)
