import pandas as pd
import numpy as np
from datetime import datetime
from config.config import DATABASE_PATH, ABBREVIATION_MICROELEMENTS_DICT
from utils.parser import Parser
import os 
import json

parser = Parser()

def parse_json(receipt_data:dict):
    """
    Parse the JSON data from a receipt and make predictions with a model.

    Args:
    receipt_data (dict): JSON data from a receipt

    Returns:
    tuple: A tuple containing a pandas DataFrame with the receipt items and a dictionary with the receipt information.
    """
    receipt_id = receipt_data.get("_id")
    receipt_doc = receipt_data["ticket"]["document"]["receipt"]

    purchase_datetime = receipt_doc.get("dateTime")
    if purchase_datetime:
        purchase_datetime = datetime.strptime(purchase_datetime, "%Y-%m-%dT%H:%M:%S").strftime("%Y-%m-%d_%H:%M:%S")
    if receipt_doc.get("ecashTotalSum", 0) == 0:
        total_sum = receipt_doc.get("cashTotalSum", 0)
    else:
        total_sum = receipt_doc.get("ecashTotalSum", 0)
    
    total_sum = float(total_sum) / 100  # to rubles

    receipt_info = {
        "receipt_id": receipt_id,
        "purchase_datetime": purchase_datetime,
        "total_sum": total_sum,
    }

    items = receipt_doc["items"]
    items = [
        {
            "name": item["name"],
            "quantity": item["quantity"],
            "sum_rub": float(item["sum"]) / 100,
        }
        for item in items
    ]

    return pd.DataFrame(items), receipt_info
 
def parse_predict(receipt_data, bert_2level_model, le, min_confidence=0.5, ):
    """
    Predicts product categories for a receipt using a BERT model.

    Args:
        receipt_data (dict): JSON data from a receipt
        bert_2level_model (BERTTwoLevelModel): A trained BERT model for product category prediction
        le (LabelEncoder): A LabelEncoder for encoding product categories
        min_confidence (float, optional): The minimal confidence for a prediction to be considered correct. Defaults to 0.5.

    Returns:
        tuple: A tuple containing a pandas DataFrame with the receipt items and their predicted categories, and a dictionary with the receipt information.
    """
    data, receipt_info = parse_json(receipt_data)
    product_names = data['name'].tolist()

    parsed_data = parser.parse_dataset(
        product_names,
        extract_cost=False,
        extract_hierarchical_number=False,
    )
    data = data.join(parsed_data)

    product_names = parsed_data['product_name'].tolist()
    predictions, confidences = bert_2level_model.predict(product_names)
    predictions = le.inverse_transform(predictions)
    data['prediction'] = predictions


    confidences = np.array(confidences)
    user_predictions = np.where(confidences >= min_confidence, predictions, 'нераспознанное')
    data['user_prediction'] = user_predictions
    data['confidence'] = confidences

    return data, receipt_info

def save_data_for_dataset(new_data, dynamic_dataset_path):
    """
    Saves new data to a dynamic dataset, which is a CSV file. It concatenates the new data with the existing data, removes duplicates, and saves the updated data back to the CSV file.

    Args:
        new_data (list of dict): A list of dictionaries, where each dictionary contains information about a product and its category.
        dynamic_dataset_path (str): The path to the dynamic dataset CSV file.
    """
    try:
        existing_data = pd.read_csv(dynamic_dataset_path, sep='|') 
    except pd.errors.EmptyDataError:
        existing_data = pd.DataFrame()
    
    new_data = pd.DataFrame(new_data)
    new_data.drop(columns=['user_prediction', 'original_entry', 'sum_rub', 'quantity'], inplace=True)
    
    updated_data = pd.concat([existing_data, new_data], ignore_index=True)
    updated_data.drop_duplicates(subset=['name'], inplace=True)
    
    updated_data.to_csv(dynamic_dataset_path, sep='|', index=False)


# ==== Sort receipts =====

def get_sorted_user_receipts(user_id, database_path):
    """
    Sorts a user's receipts by their purchase date in descending order.

    Args:
        user_id (int): The user ID to get the receipts for.
        database_path (str): The path to the database.

    Returns:
        list: A sorted list of the user's receipts.
    """
    user_json_path = os.path.join(database_path, f"user_{user_id}", "user_profile.json")
    with open(user_json_path, 'r') as f:
        user_data = json.load(f)

    receipts = user_data.get('user_purchases', [])
    sorted_receipts = sorted(receipts, key=lambda x: datetime.strptime(x['purchase_datetime'], '%Y-%m-%d_%H:%M:%S'), reverse=True)
    return sorted_receipts


# ==== Categories data processing =====

def count_product_amounts(user_id):
    """
    Counts the total amount of each product in a user's purchases.

    Args:
        user_id (int): The user ID to count the products for.

    Returns:
        tuple: A tuple containing two dictionaries. The first dictionary has the product names as keys and their total amounts as values.
        The second dictionary has the product names as keys and their sources (i.e. the products that were detected as containing the product) as values.
    """
    product_counts = {}
    undetected_categories = {}

    user_json_path = os.path.join(DATABASE_PATH, f"user_{user_id}", "user_profile.json")
    with open(user_json_path, 'r') as f:
        user_data = json.load(f)

    default_amounts = user_data.get('default_amounts', {})
    receipts = user_data.get('user_purchases', [])
    
    for purchase in receipts:
        for item in purchase['items']:
            product_name = item.get('name')
            user_prediction = item.get('user_prediction', '').strip()

            # Skip items with 'несъедобное' prediction
            if user_prediction == 'несъедобное':
                continue
            
            quantity = item.get('quantity', 1)
            if item.get('amount', 'n/a') != 'n/a':
                amount, unit = item['amount'].split()
            else:
                unit = 'n/a'
                
            default_amount_grams = default_amounts.get(user_prediction, None)
            
            if default_amount_grams:
                amount_grams = default_amount_grams * quantity
            elif item.get('amount', 'n/a') != 'n/a' and unit != 'шт':
                amount_grams = parser.convert_amount_units_to_grams(item['amount']) * quantity
            else:
                if user_prediction not in undetected_categories:
                    undetected_categories[user_prediction] = {'total_amount': 'n/a', 'sources': []}
                undetected_categories[user_prediction]['sources'].append(product_name)
                continue
            
            # Add to product_counts
            if user_prediction not in product_counts:
                product_counts[user_prediction] = {'total_amount': 0, 'sources': []}
            
            product_counts[user_prediction]['total_amount'] += amount_grams
            product_counts[user_prediction]['sources'].append((product_name, amount_grams))

    # Sort sources for each product
    for product in product_counts:
        product_counts[product]['sources'].sort(key=lambda x: x[1], reverse=True)

    # Sort product_counts by total_amount
    sorted_products = sorted(product_counts.items(), key=lambda x: x[1]['total_amount'], reverse=True)
    product_counts = dict(sorted_products)

    return product_counts, undetected_categories


# ==== Microelements data processing =====

def restructure_microelements():
    """
    Restructure the microelements table from a flat CSV to a nested dictionary of product_name -> microelement_name -> amount.

    The function reads the cleaned microelements table from the database, and for each row, it creates a nested dictionary 
    where the outer key is the product name, and the inner keys are the microelement names. The microelement names are 
    translated to their full names using the ABBREVIATION_MICROELEMENTS_DICT.

    Returns:
        dict: A nested dictionary of product_name -> microelement_name -> amount.
    """
    microelements_table = pd.read_csv(os.path.join(DATABASE_PATH, 'cleaned_microelements_table.csv'), sep='|')
    
    restructured = {}
    
    for _, row in microelements_table.iterrows():
        product_name = row['Продукт в ТХС'].lower().strip()
        product_data = {}
        
        for column in microelements_table.columns:
            if column != 'Продукт в ТХС':
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
            microelements[key] = float(value) * (amount/100)  # All values are relative to 100 grams
        except ValueError:
            pass
    return microelements

def get_microelements_data(user_id, microelements_table):
    """
    Get the microelements data for a given user ID.

    Args:
        user_id (int): The ID of the user.
        microelements_table (dict): The nested dictionary of product_name -> microelement_name -> amount.

    Returns:
        dict: A dictionary with the microelement names as keys and dictionaries with total_amount and sources as values.
    """
    
    microelements_data = {}
    product_counts, _ = count_product_amounts(user_id)
    
    for product, data in product_counts.items():
        if product in microelements_table:
            total_amount = data['total_amount']
            product_microelements = get_microelements_for_product(product, total_amount, microelements_table)
            
            for element, value in product_microelements.items():
                if element not in microelements_data:
                    microelements_data[element] = {'total_amount': 0, 'sources': []}
                
                # Only add to total and sources if the value is greater than zero
                if value > 0:
                    microelements_data[element]['total_amount'] += value
                    microelements_data[element]['sources'].append((product, value))

    # Remove elements with total_amount equal to 0
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

    # Add the rest of the elements
    for element, data in sorted_microelements:
        if element != "Энергетическая ценность":
            final_microelements[element] = data

    return final_microelements