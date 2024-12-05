import sqlite3
import os
import json
from datetime import datetime
from utils.data_processor import parse_predict
from model.model_init import initialize_model
from utils.utils import save_user_data
from config.config import DATABASE_PATH

DATABASE_FOLDER_PATH = DATABASE_PATH
DATABASE_PATH = 'data/database.db'

bert_2level_model, le = initialize_model()

# Connect to SQLite database
def create_database():
    """
    Connects to a SQLite database file and creates all the needed tables.
    :return: None
    """
    conn = sqlite3.connect(DATABASE_PATH)
    conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign keys
    cursor = conn.cursor()

    # Create tables
    cursor.execute("""
    -- main_database
    -- This table stores user's information
    CREATE TABLE main_database (
        user_id INTEGER PRIMARY KEY,
        user_name TEXT,
        registration_date DATE NOT NULL,
        add_to_history BOOL NOT NULL DEFAULT 0,
        return_excel_document BOOL NOT NULL DEFAULT 0,
        original_receipts_added INTEGER NOT NULL DEFAULT 0,
        products_added INTEGER NOT NULL DEFAULT 0,
        minimal_confidence_for_prediction REAL NOT NULL DEFAULT 0.5
    );
    """)

    cursor.execute("""
    -- default_amounts
    -- This table stores default amounts for each item for each user
    -- Composite primary key (item_name, user_id) to ensure that each item
    -- for each user has a default amount
    CREATE TABLE default_amounts (
        user_id INTEGER NOT NULL,
        item_name TEXT NOT NULL,
        item_amount_grams REAL NOT NULL,
        PRIMARY KEY (item_name, user_id),  -- Composite primary key
        FOREIGN KEY (user_id) REFERENCES main_database (user_id)
            ON DELETE CASCADE ON UPDATE CASCADE
    );
    """)

    cursor.execute("""
    -- user_purchases
    -- This table stores receipts for each user
    CREATE TABLE user_purchases (
        receipt_id TEXT PRIMARY KEY,
        user_id INTEGER NOT NULL,
        purchase_datetime DATETIME,
        total_sum REAL,
        in_history BOOL NOT NULL DEFAULT 1,
        FOREIGN KEY (user_id) REFERENCES main_database (user_id)
            ON DELETE CASCADE ON UPDATE CASCADE
    );
    """)

    cursor.execute("""
    -- receipt_items
    -- This table stores items for each receipt
    CREATE TABLE receipt_items (
        item_id INTEGER PRIMARY KEY AUTOINCREMENT,
        receipt_id TEXT NOT NULL,
        user_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        quantity INTEGER DEFAULT 0,
        sum_rub REAL DEFAULT 0.0,
        original_entry TEXT,
        percentage REAL DEFAULT 0.0,
        amount REAL DEFAULT 0.0,
        product_name TEXT,
        portion REAL DEFAULT 0.0,
        prediction TEXT,
        user_prediction TEXT,
        confidence REAL DEFAULT 0.0,
        in_history BOOL NOT NULL DEFAULT 1,
        FOREIGN KEY (receipt_id) REFERENCES user_purchases (receipt_id)
            ON DELETE CASCADE ON UPDATE CASCADE
    );
    """)

    conn.commit()
    conn.close()

def user_profile_to_db(user_profile: dict) -> None:
    """
    Inserts a user's profile into the SQLite database.
    Handles UNIQUE constraint violations gracefully.
    :param user_profile: A dictionary containing the user's profile information.
    :return: None
    """
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    try:
        # Insert into main_database
        try:
            cursor.execute("""
                INSERT INTO main_database (user_id, user_name, registration_date, add_to_history, return_excel_document)
                VALUES (:user_id, :user_name, :registration_date, :add_to_history, :return_excel_document)
            """, {
                'user_id': user_profile['user_id'],
                'user_name': user_profile.get('user_name', 'nameless'),
                'registration_date': datetime.strptime(user_profile.get('registration_date', ''), '%d-%m-%Y_%H:%M:%S').strftime('%Y-%m-%d %H:%M:%S'),
                'add_to_history': user_profile.get('add_to_history', False),
                'return_excel_document': user_profile.get('return_excel_document', False),
            })
        except sqlite3.IntegrityError as e:
            print(f"User {user_profile['user_id']} already exists in main_database: {e}")

        # Insert into default_amounts
        for default_amount in user_profile.get('default_amounts', {}).items():
            try:
                cursor.execute("""
                    INSERT INTO default_amounts (item_name, user_id, item_amount_grams)
                    VALUES (:item_name, :user_id, :item_amount_grams)
                """, {
                    'item_name': default_amount[0],
                    'user_id': user_profile['user_id'],
                    'item_amount_grams': default_amount[1],
                })
            except sqlite3.IntegrityError as e:
                print(f"Skipping default_amounts entry {default_amount} due to: {e}")

        # Insert into user_purchases
        for receipt in user_profile.get('user_purchases', []):
            try:
                # Check for duplicate receipt_id
                cursor.execute("""
                    SELECT receipt_id FROM user_purchases WHERE receipt_id = :receipt_id
                """, {'receipt_id': receipt['receipt_id']})
                if cursor.fetchone():
                    print(f"Receipt {receipt['receipt_id']} already exists. Skipping.")
                    continue

                purchase_datetime = datetime.strptime(receipt.get('purchase_datetime', ''), '%Y-%m-%d_%H:%M:%S').strftime('%Y-%m-%d %H:%M:%S')
                cursor.execute("""
                    INSERT INTO user_purchases (receipt_id, user_id, purchase_datetime, total_sum, in_history)
                    VALUES (:receipt_id, :user_id, :purchase_datetime, :total_sum, :in_history)
                """, {
                    'receipt_id': receipt['receipt_id'],
                    'user_id': user_profile['user_id'],
                    'purchase_datetime': purchase_datetime,
                    'total_sum': receipt['total_sum'],
                    'in_history': receipt.get('in_history', 1),
                })
            except sqlite3.IntegrityError as e:
                print(f"Skipping user_purchases entry {receipt['receipt_id']} due to: {e}")
                continue  # Skip items for this receipt if the receipt itself fails

            # Insert receipt items
            for item in receipt.get('items', []):
                try:
                    item = {key: None if value == 'n/a' else value for key, value in item.items()}
                    cursor.execute("""
                        INSERT INTO receipt_items (
                            receipt_id, user_id, name, quantity, sum_rub, original_entry, percentage, amount, 
                            product_name, portion, prediction, user_prediction, confidence, in_history
                        ) VALUES (
                            :receipt_id, :user_id, :name, :quantity, :sum_rub, :original_entry, :percentage, :amount, 
                            :product_name, :portion, :prediction, :user_prediction, :confidence, :in_history
                        )
                    """, {
                        'receipt_id': receipt['receipt_id'],
                        'user_id': user_profile['user_id'],
                        'name': item.get('name', None),
                        'quantity': item.get('quantity', 1),
                        'sum_rub': item.get('sum_rub', None),
                        'original_entry': item.get('original_entry', None),
                        'percentage': item.get('percentage', None),
                        'amount': item.get('amount', None),
                        'product_name': item.get('product_name', None),
                        'portion': item.get('portion', None),
                        'prediction': item.get('prediction', None),
                        'user_prediction': item.get('user_prediction', None),
                        'confidence': item.get('confidence', 0.5),
                        'in_history': receipt.get('in_history', 1),
                    })
                except sqlite3.IntegrityError as e:
                    print(f"Skipping receipt_items entry for {item.get('name', 'unknown')} due to: {e}")

        conn.commit()
    except sqlite3.Error as e:
        print(f"An error occurred while processing user profile: {e}")
        conn.rollback()
    finally:
        conn.close()



def update_user_profile_db(user_id):
    """
    Updates the original_receipts_added and products_added fields in the main_database table
    based on the current number of receipts and items associated with the user.
    :param user_id: The ID of the user to update.
    """
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    try:
        # Count receipts for the user
        cursor.execute("""
            SELECT COUNT(*) 
            FROM user_purchases 
            WHERE user_id = :user_id
        """, {'user_id': user_id})
        receipts_count = cursor.fetchone()[0]

        # Count items for the user
        cursor.execute("""
            SELECT COUNT(*) 
            FROM receipt_items 
            WHERE user_id = :user_id
        """, {'user_id': user_id})
        products_count = cursor.fetchone()[0]

        # Update the main_database table
        cursor.execute("""
            UPDATE main_database
            SET original_receipts_added = :receipts_count,
                products_added = :products_count
            WHERE user_id = :user_id
        """, {
            'receipts_count': receipts_count,
            'products_count': products_count,
            'user_id': user_id
        })

        # Commit changes
        conn.commit()
        print(f"Updated user {user_id}: receipts = {receipts_count}, products = {products_count}")

    except Exception as e:
        conn.rollback()
        print(f"Failed to update user {user_id}: {e}")

    finally:
        conn.close()


def get_user_profile(user_id):    
    user_path = os.path.join(DATABASE_FOLDER_PATH, f'user_{user_id}')

    user_profile_path = [file for file in os.listdir(user_path) if file.startswith('user_profile')][0]
    if user_profile_path is None:
        raise FileNotFoundError(f'User {user_id} profile is not found')

    with open(os.path.join(user_path, user_profile_path), 'r', encoding='utf-8') as f:
        user_profile = json.load(f)

    return user_profile

def add_receipts_to_json(user_id, file_name):
    user_folder = os.path.join(DATABASE_FOLDER_PATH, f"user_{user_id}")
    file_path = os.path.join(user_folder, file_name)
    
    user_data = get_user_profile(user_id)

    with open(file_path, 'r') as f:
        data_received = json.load(f)

    print(f"User {user_id} file {file_name} loaded")
    
    if isinstance(data_received, list):
        print(f"File {file_name} contains a list of {len(data_received)} receipts")
        for receipt_data in data_received:
            data, receipt_info = parse_predict(receipt_data, bert_2level_model, le)
            receipt_ids = [receipt.get('receipt_id') for receipt in user_data['user_purchases']]
            if not receipt_info['receipt_id'] in receipt_ids:
                save_user_data(data, receipt_info, user_id, add_to_history=False)
                print(f"Receipt {receipt_info['receipt_id']} added to user {user_id}")
            else:
                print(f"Receipt {receipt_info['receipt_id']} already exists in user {user_id}")
    else:
        print(f"File {file_name} contains a single receipt")
        data, receipt_info = parse_predict(data_received, bert_2level_model, le)
        receipt_ids = [receipt.get('receipt_id') for receipt in user_data['user_purchases']]
        if not receipt_info['receipt_id'] in receipt_ids:
            save_user_data(data, receipt_info, user_id, add_to_history=False)
            print(f"Receipt {receipt_info['receipt_id']} added to user {user_id}")
        else:
            print(f"Receipt {receipt_info['receipt_id']} already exists in user {user_id}")
