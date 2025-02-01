import pandas as pd
import numpy as np
from datetime import datetime
from config.config import DATABASE_PATH, ABBREVIATION_MICROELEMENTS_DICT
from utils.parser import Parser
from utils.db_utils import (
    get_connection, UserPurchases, ReceiptItems,
    UserSettings, DefaultAmounts
)
import os
import uuid
from utils.logger import logger

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
        # Datetime object handling
        purchase_datetime = receipt_doc.get("dateTime")
        
        if purchase_datetime and isinstance(purchase_datetime, str):
            for format_string in ["%Y-%m-%dT%H:%M:%S", '%Y-%m-%d_%H:%M:%S']:
                try:
                    purchase_datetime = datetime.strptime(purchase_datetime, format_string)
                    break # Stop if found formatting
                except:
                    pass
        elif 'localDateTime' in receipt_doc:
            purchase_datetime = datetime.strptime(receipt_doc['localDateTime'], "%Y-%m-%dT%H:%M")
        else:
            purchase_datetime = None #* Important
        
        # Tax information and company name
        retail_place = receipt_doc.get("retailPlace", None)
        retail_place_address = receipt_doc.get("retailPlaceAddress", None)
        company_name = receipt_doc.get("user", None)
        inn = receipt_doc.get("userInn", None)
        
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
        'retail_place': retail_place,
        'retail_place_address': retail_place_address,
        'company_name': company_name,
        'inn': inn
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

def save_data_for_database(items_data: pd.DataFrame, receipt_info: dict, user_id: int):
    """Save receipt and items data to database with improved transaction handling."""
    # If receipt doesn't exist, create it in a new transaction
    try:
        with get_connection() as session:
            add_to_history_bool = session.query(UserSettings.add_to_history).filter(UserSettings.user_id == user_id).scalar()
            # Start with the parent record
            receipt_purchase = UserPurchases(
                receipt_id = receipt_info['receipt_id'],
                user_id = user_id,
                purchase_datetime = receipt_info.get('purchase_datetime'),
                total_sum = receipt_info.get('total_sum'),
                in_history = add_to_history_bool,
                retail_place = receipt_info.get('retail_place'),
                retail_place_address = receipt_info.get('retail_place_address'),
                company_name = receipt_info.get('company_name'),
                inn = receipt_info.get('inn')
            )
            session.add(receipt_purchase)
        

        with get_connection() as session:
            # Prepare all items before adding anything
            receipt_items = []
            for _, item in items_data.iterrows():
                receipt_item = ReceiptItems(
                    receipt_id = receipt_info['receipt_id'],
                    user_id = user_id,
                    quantity = item.get('quantity'),
                    amount = item.get('amount'),
                    price = item.get('sum_rub'),
                    product_name = item.get('name'),
                    portion = item.get('portion'),
                    prediction = item.get('prediction'),
                    user_prediction = item.get('user_prediction'),
                    confidence = item.get('confidence'),
                    in_history = add_to_history_bool
                )
                receipt_items.append(receipt_item)
            # Add items one by one to better handle potential errors
            for item in receipt_items:
                session.add(item)
                session.flush()  # Verify each item can be added
                            
    except Exception as e:
        session.rollback()
        raise e

def get_sorted_user_receipts(user_id):
    """
    Sorts a user's receipts by their purchase date in descending order using SQLAlchemy.
    """
    with get_connection() as session:
        receipts = session.query(UserPurchases).filter(
            UserPurchases.user_id == user_id,
            UserPurchases.in_history == True
        ).order_by(UserPurchases.purchase_datetime.desc()).all()
        
        # Convert SQLAlchemy objects to dictionaries
        return [
            {
                'receipt_id': r.receipt_id,
                'user_id': r.user_id,
                'purchase_datetime': r.purchase_datetime,
                'total_sum': r.total_sum,
                'in_history': r.in_history
            }
            for r in receipts
        ]

def count_product_amounts(user_id):
    """
    Counts the total amount of each product in a user's purchases using SQLAlchemy.
    """
    with get_connection() as session:
        # Get default amounts
        default_amounts_query = session.query(DefaultAmounts)
        default_amounts = {
            row.item_name: row.item_amount 
            for row in default_amounts_query.all()
        }

        # Get all items for this user
        items_query = session.query(
            ReceiptItems.product_name,
            ReceiptItems.quantity,
            ReceiptItems.amount,
            ReceiptItems.user_prediction,
            ReceiptItems.receipt_id
        ).join(
            UserPurchases,
            UserPurchases.receipt_id == ReceiptItems.receipt_id
        ).filter(
            UserPurchases.user_id == user_id,
            UserPurchases.in_history == True
        )
        
        items_data = items_query.all()

    product_counts = {}
    undetected_categories = {}

    for row in items_data:
        product_name = row.product_name
        user_prediction = (row.user_prediction or '').strip()
        quantity = row.quantity if row.quantity is not None else 1
        amount_str = row.amount

        # Skip items predicted as несъедобное (inedible)
        if user_prediction == 'несъедобное':
            continue

        # Determine the amount in grams
        amount_grams = None
        default_amount_grams = default_amounts.get(user_prediction, None)

        if default_amount_grams is not None:
            amount_grams = default_amount_grams * quantity
        else:
            if amount_str is not None:
                parts = amount_str.split()
                if len(parts) == 2:
                    try:
                        converted_unit = parser.convert_amount_units_to_grams(amount_str)
                        if converted_unit is not None:
                            amount_grams = converted_unit * quantity
                    except Exception:
                        pass

        if amount_grams is None:
            if user_prediction not in undetected_categories:
                undetected_categories[user_prediction] = {'total_amount': 'n/a', 'sources': []}
            undetected_categories[user_prediction]['sources'].append(product_name)
            continue

        # Add to product_counts
        if user_prediction not in product_counts:
            product_counts[user_prediction] = {'total_amount': 0, 'sources': []}

        product_counts[user_prediction]['total_amount'] += amount_grams
        product_counts[user_prediction]['sources'].append((product_name, amount_grams))

    # Sort sources and products
    for product in product_counts:
        product_counts[product]['sources'].sort(key=lambda x: x[1], reverse=True)

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

def count_product_amounts_by_period(user_id, start_date, end_date):
    """
    Counts the total amount of each product in a user's purchases within a specified period.
    
    Args:
        user_id (int): The user's ID.
        start_date (datetime): The start datetime for the period.
        end_date (datetime): The end datetime for the period.
        
    Returns:
        tuple: A tuple (product_counts, undetected_categories) where product_counts is a 
               dict mapping product (user_prediction) to total amount and sources, and 
               undetected_categories contains items for which an amount could not be determined.
    """
    with get_connection() as session:
        # Get default amounts from the database
        default_amounts_query = session.query(DefaultAmounts)
        default_amounts = {row.item_name: row.item_amount for row in default_amounts_query.all()}

        # Query receipt items joined with their parent receipts filtered by purchase_datetime
        items_query = session.query(
            ReceiptItems.product_name,
            ReceiptItems.quantity,
            ReceiptItems.amount,
            ReceiptItems.user_prediction,
            ReceiptItems.receipt_id
        ).join(
            UserPurchases,
            (UserPurchases.receipt_id == ReceiptItems.receipt_id) &
            (UserPurchases.user_id == ReceiptItems.user_id)
        ).filter(
            UserPurchases.user_id == user_id,
            UserPurchases.in_history == True,
            UserPurchases.purchase_datetime >= start_date,
            UserPurchases.purchase_datetime <= end_date
        )
        items_data = items_query.all()

    product_counts = {}
    undetected_categories = {}

    # Process each retrieved item
    for row in items_data:
        product_name = row.product_name
        user_prediction = (row.user_prediction or '').strip()
        quantity = row.quantity if row.quantity is not None else 1
        amount_str = row.amount

        # Skip inedible items
        if user_prediction == 'несъедобное':
            continue

        amount_grams = None
        default_amount_grams = default_amounts.get(user_prediction, None)

        if default_amount_grams is not None:
            amount_grams = default_amount_grams * quantity
        else:
            if amount_str is not None:
                parts = amount_str.split()
                logger.debug(parts)
                if len(parts) == 2:
                    try:
                        amount_in_grams = parser.convert_amount_units_to_grams(amount_str)
                        logger.debug(f'"{amount_str}" -> {amount_in_grams}')
                        if amount_in_grams is not None:
                            amount_grams = amount_in_grams * quantity
                    except Exception:
                        pass

        if amount_grams is None:
            if user_prediction not in undetected_categories:
                undetected_categories[user_prediction] = {'total_amount': 0, 'sources': []}
            undetected_categories[user_prediction]['sources'].append(product_name)
            continue

        if user_prediction not in product_counts:
            product_counts[user_prediction] = {'total_amount': 0, 'sources': []}

        product_counts[user_prediction]['total_amount'] += amount_grams
        product_counts[user_prediction]['sources'].append((product_name, amount_grams))

    # Sort the sources for each product
    for product in product_counts:
        product_counts[product]['sources'].sort(key=lambda x: x[1], reverse=True)

    # Sort products by total_amount in descending order
    sorted_products = sorted(product_counts.items(), key=lambda x: x[1]['total_amount'], reverse=True)
    product_counts = dict(sorted_products)

    return product_counts, undetected_categories


def get_microelements_data_by_period(user_id, start_date, end_date, microelements_table):
    """
    Computes the microelements data for a user's purchases within a specified period.
    
    This function uses the product counts (in grams) from receipts in the date range 
    and applies the microelements data per 100 grams from the provided table.
    
    Args:
        user_id (int): The user's ID.
        start_date (datetime): The start datetime for the period.
        end_date (datetime): The end datetime for the period.
        microelements_table (dict): The nested dictionary (product_name -> microelement_name -> amount)
                                    typically obtained via restructure_microelements().
                                    
    Returns:
        dict: A dictionary with microelement names as keys and dictionaries (with total_amount and sources) as values.
        Also returns undetected_categories from the counting step.
    """
    product_counts, undetected_categories = count_product_amounts_by_period(user_id, start_date, end_date)

    microelements_data = {}

    for product, data in product_counts.items():
        if product in microelements_table:
            total_amount = data['total_amount']
            # Calculate microelements for the product
            product_microelements = get_microelements_for_product(product, total_amount, microelements_table)

            for element, value in product_microelements.items():
                if element not in microelements_data:
                    microelements_data[element] = {'total_amount': 0, 'sources': []}
                # Only add if the value is positive
                if value > 0:
                    microelements_data[element]['total_amount'] += value
                    microelements_data[element]['sources'].append((product, value))

    # Remove microelements with non-positive total_amount
    microelements_data = {k: v for k, v in microelements_data.items() if v['total_amount'] > 0}

    # Sort sources for each microelement
    for element in microelements_data:
        microelements_data[element]['sources'].sort(key=lambda x: x[1], reverse=True)

    # Sort microelements by total_amount in descending order
    sorted_microelements = sorted(microelements_data.items(), key=lambda x: x[1]['total_amount'], reverse=True)

    # Place "Энергетическая ценность" first if present
    final_microelements = {}
    for element, data in sorted_microelements:
        if element == "Энергетическая ценность":
            final_microelements[element] = data
            break
    for element, data in sorted_microelements:
        if element != "Энергетическая ценность":
            final_microelements[element] = data

    return final_microelements, undetected_categories

if __name__ == '__main__':
    from datetime import datetime
    # Define your period
    start = datetime(2024, 11, 26)
    end = datetime(2024, 11, 27)
    # Assuming you have restructured microelements table:
    microelements_table = restructure_microelements()
    micro_data, undetected = get_microelements_data_by_period(968466884, start, end, microelements_table)
    print(micro_data, undetected)
