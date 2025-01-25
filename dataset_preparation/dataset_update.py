from utils.parser import Parser
import pandas as pd
import os
from config.config import DATASETS_FOLDER
from utils.db_utils import db_connection

def load_dynamic_dataset(saving_path:str = None):
    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT *
            FROM (
                SELECT
                    *,
                    ROW_NUMBER() OVER (PARTITION BY name ORDER BY name) AS rn
                FROM receipt_items
            ) sub
            WHERE rn = 1
            ORDER BY name
            """
        )
        all_items = cursor.fetchall()
    df = pd.DataFrame([dict(row) for row in all_items])
    df.drop(columns=['item_id', 'receipt_id', 'user_id', 'quantity'], inplace=True)
    df = df[['original_entry', 'percentage', 'amount', 'product_name', 'name', 'prediction', 'portion']]
    df.rename(columns={'prediction': 'clear_name', 'name':'original_name'}, inplace=True)
    if saving_path:
        df.to_csv(saving_path, sep='|', index=False)
    return df

def update_dataset(new_data_path: str, main_data_path: str, updated_data_path: str):
    """Updates the main dataset with new data, ensuring no duplicates."""
    main_data = pd.read_csv(main_data_path, sep='|')
    new_data = pd.read_csv(new_data_path, sep='|')

    updated_data = pd.concat([main_data, new_data], axis=0, ignore_index=True)
    # Remove duplicates based on 'name' column
    updated_data.drop_duplicates(subset=['original_name'], keep='first', inplace=True)
    
    # Reset index after dropping duplicates
    updated_data.reset_index(drop=True, inplace=True)
    
    updated_data.to_csv(updated_data_path, sep='|', index=False)

MAIN_DATASET_PATH = os.path.join(DATASETS_FOLDER, 'data_nov28_combined_parsed.csv')
NEW_DATASET_PATH = os.path.join(DATASETS_FOLDER, 'dynamic_dataset_not_reviewed.csv')
UPDATED_DATASET_PATH = os.path.join(DATASETS_FOLDER, 'data_nov28_updated.csv')

assert os.path.exists(MAIN_DATASET_PATH), 'Main dataset not found'

if not os.path.exists(NEW_DATASET_PATH):
    all_items = load_dynamic_dataset(NEW_DATASET_PATH)
    print(f'Loaded {len(all_items)} new entries')
else:
    inp = input('New dataset is already present, do you want to overwrite it? (THIS ACTION IS IRREVERSIBLE) (y/n): ')
    
    if inp == 'y':
        all_items = load_dynamic_dataset(NEW_DATASET_PATH)
        print(f'Loaded {len(all_items)} new entries')
    else:
        print('New dataset wasn\'t overwritten')

if not os.path.exists(UPDATED_DATASET_PATH):
    inp = input('Do you want to update the dataset? (MAKE SURE YOU REVIEWED IT FIRST!) (y/n): ')
    
    if inp=='y':
        update_dataset(NEW_DATASET_PATH, MAIN_DATASET_PATH, UPDATED_DATASET_PATH)
        print('Dataset was updated')
    else:
        print('Dataset was not updated')