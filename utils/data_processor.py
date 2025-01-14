import pandas as pd
import numpy as np
from datetime import datetime
from config.config import DATABASE_PATH, DATABASE_FILE_PATH, ABBREVIATION_MICROELEMENTS_DICT
from utils.parser import Parser
from utils.db_utils import db_connection
import os
import json
import sqlite3
from contextlib import contextmanager
import uuid

parser = Parser()

def parse_json(receipt_data: dict):
    """
    Parse the JSON data from a receipt and make predictions with a model.

    Args:
        receipt_data (dict): JSON data from a receipt

    Returns:
        tuple: A tuple containing a pandas DataFrame with the receipt items and a dictionary with the receipt information.
    """
    # Generate UUID first as fallback
    receipt_id = str(uuid.uuid4())
    
    try:
        # Try to get the _id if it exists, otherwise keep the UUID
        if "_id" in receipt_data:
            receipt_id = receipt_data["_id"]
            
        if receipt_data.get('ticket'):
            receipt_doc = receipt_data["ticket"]["document"]["receipt"]
        else:
            receipt_doc = receipt_data
        purchase_datetime = receipt_doc.get("dateTime")
        
        if purchase_datetime and isinstance(purchase_datetime, str):
            for format_string in ["%Y-%m-%dT%H:%M:%S"]:
                try:
                    purchase_datetime = datetime.strptime(purchase_datetime, format_string).strftime("%Y-%m-%d_%H:%M:%S")
                except:
                    pass
        elif 'localDateTime' in receipt_doc:
            purchase_datetime = datetime.strptime(receipt_doc['localDateTime'], "%Y-%m-%dT%H:%M").strftime("%Y-%m-%d_%H:%M:%S")
        else:
            purchase_datetime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            
        total_sum = receipt_doc.get("ecashTotalSum", receipt_doc.get("cashTotalSum", 0))
        total_sum = float(total_sum) / 100  # to rubles

        items = receipt_doc["items"]
        items = [
            {
                "name": item["name"],
                "quantity": item["quantity"],
                "sum_rub": float(item["sum"]) / 100,
            }
            for item in items
        ]

    except Exception as e:
        raise ValueError(f"Error parsing receipt data: {str(e)}")

    receipt_info = {
        "receipt_id": receipt_id,
        "purchase_datetime": purchase_datetime,
        "total_sum": total_sum,
    }

    return pd.DataFrame(items), receipt_info

def predict_product_categories(items_data, bert_2level_model, le, min_confidence=0.5):
    """
    Predicts product categories for a receipt using a BERT model.

    Args:
        items_data (dict): JSON items_data from a receipt
        bert_2level_model (BERTTwoLevelModel): A trained BERT model for product category prediction
        le (LabelEncoder): A LabelEncoder for encoding product categories
        min_confidence (float, optional): The minimal confidence for a prediction to be considered correct. Defaults to 0.5.

    Returns:
        tuple: A tuple containing a pandas DataFrame with the receipt items and their predicted categories.
    """
    product_names = items_data['name'].tolist()

    parsed_data = parser.parse_dataset(
        product_names,
        extract_cost=False,
        extract_hierarchical_number=False,
    )
    items_data = items_data.join(parsed_data)

    product_names = parsed_data['product_name'].tolist()
    predictions, confidences = bert_2level_model.predict(product_names)
    predictions = le.inverse_transform(predictions)
    items_data['prediction'] = predictions

    confidences = np.array(confidences)
    user_predictions = np.where(confidences >= min_confidence, predictions, 'нераспознанное')
    items_data['user_prediction'] = user_predictions
    items_data['confidence'] = confidences

    return items_data

def save_data_for_database(new_data, receipt_info, user_id):
    """
    Saves the processed receipt data into the SQLite database.

    Args:
        new_data (DataFrame): A pandas DataFrame containing receipt items with predictions.
        receipt_info (dict): A dictionary containing receipt metadata.
        user_id (int, optional): The user ID associated with the receipt. Defaults to None.
    """
    # Use context manager for DB connection
    with db_connection() as conn:
        cursor = conn.cursor()

        user_data_row = cursor.execute("SELECT * FROM main_database WHERE user_id = ?", (user_id,)).fetchone()
        user_data = dict(user_data_row) if user_data_row else None

        if user_data is None:
            print(f"User with ID {user_id} not found in the database.")
            return

        try:
            # Insert the receipt into the user_purchases table
            try:
                # We insert a new receipt. If this receipt_id is duplicate, we skip it.
                cursor.execute("""
                    INSERT INTO user_purchases (receipt_id, user_id, purchase_datetime, total_sum, in_history)
                    VALUES (:receipt_id, :user_id, :purchase_datetime, :total_sum, :in_history)
                """, {
                    'receipt_id': receipt_info['receipt_id'],
                    'user_id': user_id,
                    'purchase_datetime': datetime.strptime(receipt_info['purchase_datetime'], '%Y-%m-%d_%H:%M:%S').strftime('%Y-%m-%d %H:%M:%S'),
                    'total_sum': receipt_info['total_sum'],
                    'in_history': user_data.get('add_to_history', False)
                })
            except sqlite3.IntegrityError as e:
                print(f"Skipping duplicate receipt {receipt_info['receipt_id']} due to: {e}")
                return

            # Insert receipt items into the receipt_items table
            for _, row in new_data.iterrows():
                try:
                    cursor.execute("""
                        INSERT INTO receipt_items (
                            receipt_id, user_id, name, quantity, sum_rub, original_entry, percentage, amount,
                            product_name, portion, prediction, user_prediction, confidence, in_history
                        ) VALUES (
                            :receipt_id, :user_id, :name, :quantity, :sum_rub, :original_entry, :percentage, :amount,
                            :product_name, :portion, :prediction, :user_prediction, :confidence, :in_history
                        )
                    """, {
                        'receipt_id': receipt_info['receipt_id'],
                        'user_id': user_id,
                        'name': row.get('name'),
                        'quantity': row.get('quantity', 1),
                        'sum_rub': row.get('sum_rub', 0),
                        'original_entry': row.get('original_entry', None),
                        'percentage': row.get('percentage', None),
                        'amount': row.get('amount', None),
                        'product_name': row.get('product_name', None),
                        'portion': row.get('portion', None),
                        'prediction': row.get('prediction', None),
                        'user_prediction': row.get('user_prediction', None),
                        'confidence': row.get('confidence', 0.0),
                        'in_history': user_data.get('add_to_history', False)
                    })
                except sqlite3.IntegrityError as e:
                    print(f"Skipping duplicate item {row['name']} for receipt {receipt_info['receipt_id']} due to: {e}")

            conn.commit()
            print(f"Receipt {receipt_info['receipt_id']} saved successfully.")
        except sqlite3.Error as e:
            print(f"An error occurred while saving receipt {receipt_info['receipt_id']}: {e}")
            conn.rollback()

def get_sorted_user_receipts(user_id):
    """
    Sorts a user's receipts by their purchase date in descending order from the SQLite database.

    Args:
        user_id (int): The user ID to get the receipts for.

    Returns:
        list: A sorted list of the user's receipts, each represented as a dictionary.
    """
    # Using context manager for DB connection
    with db_connection() as conn:
        cursor = conn.cursor()
        # Fetch receipts for the user, ordered by purchase_datetime descending
        cursor.execute("""
            SELECT receipt_id, user_id, purchase_datetime, total_sum, in_history
            FROM user_purchases
            WHERE user_id = ? AND in_history = 1
            ORDER BY purchase_datetime DESC
        """, (user_id,))

        rows = cursor.fetchall()

    # Convert sqlite3.Row objects to dictionaries
    receipts = [dict(row) for row in rows]
    return receipts

def count_product_amounts(user_id):
    """
    Counts the total amount of each product in a user's purchases from the SQLite database.

    Args:
        user_id (int): The user ID to count the products for.

    Returns:
        tuple: (product_counts, undetected_categories)
               product_counts: { user_prediction: { 'total_amount': float, 'sources': [(product_name, amount), ...] }, ... }
               undetected_categories: { user_prediction: { 'total_amount': 'n/a', 'sources': [product_name, ...] }, ... }

    This function:
    - Retrieves default_amounts for the user to determine standard weights for certain products.
    - Joins user_purchases and receipt_items to get all items purchased by the user.
    - Attempts to determine amount in grams either from the default_amounts table or by parsing the item amount string.
    - If it cannot determine the amount in grams, it adds the product to undetected_categories.
    """
    with db_connection() as conn:
        cursor = conn.cursor()

        # Retrieve default_amounts for this user
        # Maps user_prediction categories to a default amount in grams, if available
        cursor.execute("""
            SELECT item_name, item_amount_grams
            FROM default_amounts
            WHERE user_id = ?
        """, (user_id,))
        default_amounts_data = cursor.fetchall()
        default_amounts = {row['item_name']: row['item_amount_grams'] for row in default_amounts_data}

        # Retrieve all items for this user by joining user_purchases and receipt_items
        # This gives us item details (name, quantity, amount, user_prediction) along with receipt_id
        cursor.execute("""
            SELECT 
                ri.name, 
                ri.quantity, 
                ri.amount, 
                ri.user_prediction, 
                up.receipt_id                
            FROM user_purchases up
            JOIN receipt_items ri ON up.receipt_id = ri.receipt_id
            WHERE up.user_id = ? AND up.in_history = 1
        """, (user_id,))
        items_data = cursor.fetchall()

    product_counts = {}
    undetected_categories = {}

    for row in items_data:
        product_name = row['name']
        user_prediction = (row['user_prediction'] or '').strip()
        quantity = row['quantity'] if row['quantity'] is not None else 1
        amount_str = row['amount']  # May be None

        # Skip items predicted as несъедобное (inedible)
        if user_prediction == 'несъедобное':
            continue

        # Determine the amount in grams
        amount_grams = None
        default_amount_grams = default_amounts.get(user_prediction, None)

        if default_amount_grams is not None:
            # Use default amount
            amount_grams = default_amount_grams * quantity
        else:
            # Try to parse amount if available
            if amount_str is not None:
                # Example format: "1 кг", "230 мл"
                parts = amount_str.split()
                if len(parts) == 2:
                    try:
                        converted = parser.convert_amount_units_to_grams(amount_str)
                        if converted is not None:
                            amount_grams = converted * quantity
                        # If converted is None, we can't determine the amount_grams
                    except Exception:
                        # If parsing or conversion fails, amount_grams stays None
                        pass

        if amount_grams is None:
            # Could not determine amount_grams
            if user_prediction not in undetected_categories:
                undetected_categories[user_prediction] = {'total_amount': 'n/a', 'sources': []}
            undetected_categories[user_prediction]['sources'].append(product_name)
            continue

        # Add to product_counts
        if user_prediction not in product_counts:
            product_counts[user_prediction] = {'total_amount': 0, 'sources': []}

        product_counts[user_prediction]['total_amount'] += amount_grams
        product_counts[user_prediction]['sources'].append((product_name, amount_grams))

    # Sort sources for each product by amount descending
    for product in product_counts:
        product_counts[product]['sources'].sort(key=lambda x: x[1], reverse=True)

    # Sort product_counts by total_amount descending
    sorted_products = sorted(product_counts.items(), key=lambda x: x[1]['total_amount'], reverse=True)
    product_counts = dict(sorted_products)

    return product_counts, undetected_categories

def restructure_microelements():
    """
    Restructure the microelements table from a flat CSV to a nested dictionary of product_name -> microelement_name -> amount.

    The function reads the cleaned microelements table and creates a nested dictionary 
    where the outer key is the product name, and the inner keys are the microelement names. 
    Microelement names are translated using ABBREVIATION_MICROELEMENTS_DICT.
    """
    microelements_table = pd.read_csv(os.path.join(DATABASE_PATH, 'cleaned_microelements_table.csv'), sep='|')

    restructured = {}
    for _, row in microelements_table.iterrows():
        product_name = row['Продукт в ТХС'].lower().strip()
        product_data = {}

        for column in microelements_table.columns:
            if column.strip() != 'Продукт в ТХС':
                # Translate the column name if it's in the dictionary
                translated_column = ABBREVIATION_MICROELEMENTS_DICT.get(column.strip(), column.strip())
                product_data[translated_column] = row[column]

        restructured[product_name] = product_data

    return restructured

def get_microelements_for_product(product_name, amount, microelements_table):
    """
    Get the microelements for a given product name and amount.

    Args:
        product_name (str): The name of the product.
        amount (int): The amount of the product in grams.
        microelements_table (dict): The nested dictionary of product_name -> microelement_name -> amount.

    Returns:
        dict: A dictionary with the microelement names as keys and the amounts in grams as values.
    """
    microelements = microelements_table.get(product_name, {}).copy()  # Create a copy
    for key, value in microelements.items():
        try:
            microelements[key] = float(value) * (amount / 100)  # All values are relative to 100 grams
        except ValueError:
            pass
    return microelements

def get_microelements_data(user_id, microelements_table):
    """
    Get the microelements data for a given user ID from the database-driven product counts.

    Args:
        user_id (int): The ID of the user.
        microelements_table (dict): The nested dictionary of product_name -> microelement_name -> amount 
                                    returned by restructure_microelements().

    Returns:
        dict: A dictionary with the microelement names as keys and dictionaries with total_amount and sources as values.
    """
    product_counts, _ = count_product_amounts(user_id)

    microelements_data = {}

    for product, data in product_counts.items():
        if product in microelements_table:
            total_amount = data['total_amount']
            product_microelements = get_microelements_for_product(product, total_amount, microelements_table)

            for element, value in product_microelements.items():
                if element not in microelements_data:
                    microelements_data[element] = {'total_amount': 0, 'sources': []}

                # Only add if the value is greater than zero
                if value > 0:
                    microelements_data[element]['total_amount'] += value
                    microelements_data[element]['sources'].append((product, value))

    # Remove elements with total_amount <= 0
    microelements_data = {k: v for k, v in microelements_data.items() if v['total_amount'] > 0}

    # Sort sources for each microelement
    for element in microelements_data:
        microelements_data[element]['sources'].sort(key=lambda x: x[1], reverse=True)

    # Sort microelements_data by total_amount in descending order
    sorted_microelements = sorted(microelements_data.items(), key=lambda x: x[1]['total_amount'], reverse=True)

    # Put "Энергетическая ценность" first if it exists
    final_microelements = {}
    for element, data in sorted_microelements:
        if element == "Энергетическая ценность":
            final_microelements[element] = data
            break

    # Add the rest of the elements after "Энергетическая ценность"
    for element, data in sorted_microelements:
        if element != "Энергетическая ценность":
            final_microelements[element] = data

    return final_microelements
