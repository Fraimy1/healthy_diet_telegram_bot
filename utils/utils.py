import os
import json
from datetime import datetime
import re
from config.config import DATABASE_PATH, DATABASE_FILE_PATH
from utils.db_utils import (
    get_connection, Users, UserSettings,
    UserPurchases, ReceiptItems
)  # Updated imports
from sqlalchemy import func
from aiogram.types import (
    InlineKeyboardMarkup,
    InlineKeyboardButton
)

def create_user_profile(user_id: int, username: str) -> None:
    """Creates a new user profile using SQLAlchemy with default values."""
    username = sanitize_username(username) if username else None
    registration_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    with get_connection() as session:
        # Check if user exists
        existing_user = session.query(Users).filter_by(user_id=user_id).first()
        if existing_user:
            print(f"User with ID {user_id} already exists in the database. No changes made.")
            return

        # Create new user
        new_user = Users(
            user_id=user_id,
            user_name=username,
            registration_date=registration_date,
            original_receipts_added=0,
            products_added=0,
            household_id=None
        )

        # Create user settings
        new_settings = UserSettings(
            user_id=user_id,
            add_to_history=True,
            minimal_prediction_confidence=0.5,
            return_excel_document=False
        )

        session.add(new_user)
        session.add(new_settings)
        print(f"User profile for user_id {user_id} created successfully in the database.")

def update_user_contributions(user_id: int):
    """Update user contribution counts using SQLAlchemy."""
    with get_connection() as session:
        # Get all user receipts ordered by datetime
        user_receipts_subq = session.query(
            UserPurchases.receipt_id,
            func.min(UserPurchases.purchase_datetime).label('first_purchase_time')
        ).group_by(
            UserPurchases.receipt_id
        ).subquery()

        # Find receipts where this user was first to add them
        unique_receipts = session.query(
            UserPurchases.receipt_id
        ).join(
            user_receipts_subq,
            (UserPurchases.receipt_id == user_receipts_subq.c.receipt_id) &
            (UserPurchases.purchase_datetime == user_receipts_subq.c.first_purchase_time)
        ).filter(
            UserPurchases.user_id == user_id
        ).distinct().all()

        # Changed from dict access to tuple access since SQLAlchemy returns tuples
        unique_receipt_ids = [r[0] for r in unique_receipts]

        # Count distinct products from unique receipts
        products_count = 0
        if unique_receipt_ids:
            products_count = session.query(func.count(ReceiptItems.item_id)).filter(
                ReceiptItems.user_id == user_id,
                ReceiptItems.receipt_id.in_(unique_receipt_ids)
            ).scalar() or 0

        # Update user record
        user = session.query(Users).filter_by(user_id=user_id).first()
        if user:
            user.original_receipts_added = len(unique_receipt_ids)
            user.products_added = products_count

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

