import json
from config.config import DATABASE_PATH
import os

def initialize_database():
    """
    Initialize the database by loading the database information and unique receipt IDs from the configuration file.

    Returns:
        database_info (dict): A dictionary containing database information.
        unique_receipt_ids (set): A set of unique receipt IDs.
    """
    with open(os.path.join(DATABASE_PATH, 'database_info.json'), 'r') as f:
        database_info = json.load(f)

    unique_receipt_ids = set(database_info['unique_receipt_ids'])

    return database_info, unique_receipt_ids

