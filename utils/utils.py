import os
import json
from datetime import datetime
import re
import sqlite3
from config.config import DATABASE_PATH, DATABASE_FILE_PATH
from utils.db_utils import db_connection
from aiogram.types import (
InlineKeyboardMarkup,
InlineKeyboardButton
)

def create_user_profile(user_id: int, username: str) -> None:
    """
    Creates a new user profile in the SQLite database with default values.
    Adds a try-except block to handle the case where the user already exists.

    :param user_id: The ID of the user to create a profile for.
    :param username: The username of the user, if available.
    :return: None
    """

    username = sanitize_username(username) if username else 'nameless'
    # Prepare user data
    user_data = {
        'user_id': user_id,
        'user_name': username,
        'registration_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'original_receipts_added': 0,
        'products_added': 0,
        'minimal_confidence_for_prediction': 0.5,
        'add_to_history': True,
        'return_excel_document': False,
        'default_amounts': {
            "яйцо": 700,
            "яйцо целое": 700,
            "чай зеленый байховый": 150
        },
        # Currently no purchases or items at creation time
        'user_purchases': []
    }

    with db_connection() as conn:
        cursor = conn.cursor()

        # Insert the user into main_database
        try:
            cursor.execute("""
                INSERT INTO main_database (
                    user_id, user_name, registration_date, add_to_history, return_excel_document,
                    original_receipts_added, products_added, minimal_confidence_for_prediction
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_data['user_id'],
                user_data['user_name'],
                user_data['registration_date'],
                user_data['add_to_history'],
                user_data['return_excel_document'],
                user_data['original_receipts_added'],
                user_data['products_added'],
                float(user_data['minimal_confidence_for_prediction'])
            ))
        except sqlite3.IntegrityError:
            # This occurs if the user_id already exists in main_database.
            print(f"User with ID {user_id} already exists in the database. No changes made.")
            return

        # Insert default amounts since main_database insert succeeded
        for item_name, amount_grams in user_data['default_amounts'].items():
            cursor.execute("""
                INSERT INTO default_amounts (user_id, item_name, item_amount_grams)
                VALUES (?, ?, ?)
            """, (
                user_data['user_id'],
                item_name,
                float(amount_grams)
            ))

        # user_purchases and receipt_items are empty at creation, so nothing to insert.

        # Commit the changes once all inserts are done
        conn.commit()

    print(f"User profile for user_id {user_id} created successfully in the database.")

def update_user_contributions(user_id: int, conn: sqlite3.Connection):
    cursor = conn.cursor()

    # First, get all receipts for the user with their timestamps
    cursor.execute("""
        SELECT receipt_id, purchase_datetime
        FROM user_purchases
        WHERE user_id = ?
    """, (user_id,))
    user_receipts = cursor.fetchall()

    if not user_receipts:
        # No receipts, just update to zero
        cursor.execute("""
            UPDATE main_database 
            SET original_receipts_added = 0, products_added = 0
            WHERE user_id = ?
        """, (user_id,))
        return

    unique_receipts = []
    for row in user_receipts:
        user_receipt_id = row['receipt_id']
        user_purchase_time = row['purchase_datetime']

        # Find the earliest user who added this receipt
        cursor.execute("""
            SELECT user_id, purchase_datetime
            FROM user_purchases
            WHERE receipt_id = ?
            ORDER BY purchase_datetime ASC
            LIMIT 1
        """, (user_receipt_id,))
        earliest = cursor.fetchone()
        
        # If current user is the earliest contributor for this receipt
        if earliest and earliest['user_id'] == user_id:
            unique_receipts.append(user_receipt_id)

    # original_receipts_added is the count of unique receipts
    original_receipts_added = len(unique_receipts)

    # Count products from these unique receipts
    # We'll assume that "products_added" means counting all distinct items from these receipts
    # If you need a different logic, adjust accordingly
    if unique_receipts:
        query = f"""
            SELECT COUNT(*) as product_count
            FROM receipt_items
            WHERE user_id = ?
            AND receipt_id IN ({','.join(['?']*len(unique_receipts))})
        """
        params = [user_id] + unique_receipts
        cursor.execute(query, params)
        products_count = cursor.fetchone()['product_count']
    else:
        products_count = 0

    # Update the main_database with new values
    cursor.execute("""
        UPDATE main_database 
        SET original_receipts_added = ?, products_added = ?
        WHERE user_id = ?
    """, (original_receipts_added, products_count, user_id))

def sanitize_username(username):
    """
    Sanitizes a username for use in the database.
    """
    return re.sub(r'[^\w\-]', '_', username)[:50]

def create_back_button(text="Назад", callback_data="back"):
    """
    Creates a back button with customizable text and callback data.
    
    :param text: The text to display on the button (default: "Назад")
    :param callback_data: The callback data for the button (default: "back")
    :return: InlineKeyboardMarkup with a single back button
    """
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text=text, callback_data=callback_data),
            ]
        ]
    )  
    return keyboard

