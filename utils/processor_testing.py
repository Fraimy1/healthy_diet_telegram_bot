import json
import sqlite3
import pytest
from config.config import DATABASE_FILE_PATH
from utils.data_processor import parse_json, predict_product_categories, save_data_for_database
from model.model_init import initialize_model

@pytest.fixture(scope="module")
def test_data():
    # Setup: Load test data from JSON which now contains a list of dictionaries
    test_file_path = 'data/database/user_968466884/06_12_2024_02_06_357157794149078618685.json'
    with open(test_file_path, 'r', encoding='utf-8') as f:
        data_received = json.load(f)  # This is now a list of dicts

    # Initialize the model and label encoder once
    bert_2level_model, le = initialize_model()

    test_data_list = []

    # Loop over each receipt dictionary in the JSON list
    for single_receipt_dict in data_received:
        # Parse the receipt data
        receipt_data, receipt_info = parse_json(single_receipt_dict)

        # Predict product categories
        items_data = predict_product_categories(receipt_data, bert_2level_model, le)

        # Append the parsed and processed data for each receipt to the list
        test_data_list.append({
            'items_data': items_data,
            'receipt_info': receipt_info,
            'user_id': 968466884
        })

    return test_data_list

def test_save_and_retrieve_data(test_data):
    # test_data is now a list of receipt data sets
    for data_set in test_data:
        items_data = data_set['items_data']
        receipt_info = data_set['receipt_info']
        user_id = data_set['user_id']

        # Attempt to save data into the database
        save_data_for_database(items_data, receipt_info, user_id)

        # Check that the user, receipt, and items are now in the database
        conn = sqlite3.connect(DATABASE_FILE_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Verify user exists in main_database
        user_data = cursor.execute("SELECT * FROM main_database WHERE user_id = ?", (user_id,)).fetchone()
        assert user_data is not None, f"User with ID {user_id} not found in the database after saving data."
        assert user_data['user_id'] == user_id, "User ID does not match expected value."

        # Verify the receipt was inserted
        saved_receipt = cursor.execute("SELECT * FROM user_purchases WHERE receipt_id = ?", (receipt_info['receipt_id'],)).fetchone()
        assert saved_receipt is not None, f"Receipt {receipt_info['receipt_id']} not found in user_purchases."
        assert saved_receipt['total_sum'] == receipt_info['total_sum'], "The total_sum in the database does not match the input data."

        # Verify at least one item is inserted for this receipt
        saved_items = cursor.execute("SELECT * FROM receipt_items WHERE receipt_id = ?", (receipt_info['receipt_id'],)).fetchall()
        assert len(saved_items) > 0, "No items found in receipt_items for the saved receipt."

        conn.close()
        print(f"Data verified successfully in the database for receipt {receipt_info['receipt_id']}.")
