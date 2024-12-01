import os
import json
from datetime import datetime
import re
from config.config import DATABASE_PATH
from aiogram.types import (
InlineKeyboardMarkup,
InlineKeyboardButton
)

def save_user_data(new_data, receipt_info, user_id):
    """Save a user's purchase data to their profile file."""
    path_to_json = os.path.join(DATABASE_PATH, f'user_{user_id}', f'user_profile.json')
    with open(path_to_json, 'r') as file:
        user_data = json.load(file)
        
    receipt_items = new_data.fillna('n/a').to_dict(orient='records')
    receipt_info['items'] = receipt_items
    
    user_data['user_purchases'].append(receipt_info)
    
    with open(path_to_json, 'w', encoding='utf-8') as file:
        json.dump(user_data, file, ensure_ascii=False, indent=4)

def create_user_file(user_id: int, username: str) -> None:
    """Create a new user profile file with default values."""
    user_folder = os.path.join(DATABASE_PATH, f'user_{user_id}')
    user_file = os.path.join(user_folder, 'user_profile.json')
    user_data = {
        'user_id': user_id,
        'user_name': username,
        'registration_date': datetime.now().strftime('%d-%m-%Y_%H:%M:%S'),
        'user_purchases': [],
        'original_receipts_added': 0,
        'products_added': 0,
        'minimal_confidence_for_prediction': 50,
        'add_to_history': True,
        'return_excel_document': False,
        "default_amounts": {
        "яйцо": 700,
        "яйцо целое": 700,
        "чай зеленый байховый": 150
    }
    }
    with open(user_file, 'w', encoding='utf-8') as file:
        json.dump(user_data, file, ensure_ascii=False, indent=4)

def sanitize_username(username):
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

