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
    Adds entries in both users and user_settings tables.
    """
    username = sanitize_username(username) if username else None
    registration_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    with db_connection() as conn:
        cursor = conn.cursor()

        try:
            # Insert into users table
            cursor.execute("""
                INSERT INTO users (
                    user_id, user_name, registration_date,
                    original_receipts_added, products_added, household_id
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                user_id,
                username,
                registration_date,
                0,  # original_receipts_added
                0,  # products_added
                None  # household_id
            ))

            # Insert into user_settings table
            cursor.execute("""
                INSERT INTO user_settings (
                    user_id, add_to_history, minimal_prediction_confidence
                ) VALUES (?, ?, ?)
            """, (
                user_id,
                True,  # add_to_history default
                0.5    # minimal_prediction_confidence default
            ))

            conn.commit()
            print(f"User profile for user_id {user_id} created successfully in the database.")

        except sqlite3.IntegrityError:
            print(f"User with ID {user_id} already exists in the database. No changes made.")
            return

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
            UPDATE users 
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

    # Update the users table with new values
    cursor.execute("""
        UPDATE users 
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

