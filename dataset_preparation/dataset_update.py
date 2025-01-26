from utils.parser import Parser
import pandas as pd
import os
from config.config import DATASETS_FOLDER
from utils.db_utils import get_connection, ReceiptItems  # Updated imports
from sqlalchemy import func

def load_dynamic_dataset(saving_path:str = None):
    """Load unique receipt items using SQLAlchemy."""
    with get_connection() as session:
        # Get unique items based on name, using window function in subquery
        subquery = session.query(
            ReceiptItems,
            func.row_number().over(
                partition_by=ReceiptItems.product_name,
                order_by=ReceiptItems.product_name
            ).label('rn')
        ).subquery()

        # Query only first occurrence of each name
        all_items = session.query(subquery).filter(
            subquery.c.rn == 1
        ).order_by(subquery.c.product_name).all()

        # Convert to DataFrame and process
        df = pd.DataFrame([{
            'original_entry': item.original_entry,
            'percentage': item.percentage,
            'amount': item.amount,
            'product_name': item.product_name,
            'name': item.product_name,  # original_name
            'prediction': item.prediction,  # clear_name
            'portion': item.portion
        } for item in all_items])

        # Rename columns to match expected format
        df.rename(columns={
            'prediction': 'clear_name',
            'name': 'original_name'
        }, inplace=True)

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