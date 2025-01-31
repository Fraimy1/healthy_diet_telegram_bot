from aiogram import F
from aiogram.types import Message, CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup
import json
from tempfile import NamedTemporaryFile

from bot.bot_init import bot, dp, bert_2level_model, le
from utils.db_utils import get_connection, UserSettings, UserPurchases  # Updated import
from utils.utils import create_back_button, update_user_contributions
from bot.handlers.menu_utils import show_menu_to_user
from utils.logger import logger
from utils.data_processor import parse_json, predict_product_categories, save_data_for_database
from datetime import datetime

# In-Memory Storage for pending receipt IDs
# pending_receipts = {}

async def process_receipt(receipt_data, user_id, idx_info=""):
    """
    Processes a receipt and updates the database accordingly.

    The function first tries to extract relevant information from the receipt data using the `predict_product_categories` function. 
    It then saves the receipt data to a file and checks if the receipt is unique. 
    If the receipt is unique, it adds the receipt data to the user's history and the dataset. 
    If the receipt is not unique, it asks the user whether they want to add the receipt to their history or not.

    Args:
        receipt_data (dict): A dictionary containing the receipt data.
        user_id (int): The user ID of the user who uploaded the receipt.

    Returns:
        None
    """
    receipt_id = None
    
    try:
        with get_connection() as session:
            user_settings = session.query(UserSettings).filter(UserSettings.user_id == user_id).first()
            min_confidence = user_settings.minimal_prediction_confidence
            add_to_history_bool = user_settings.add_to_history
            unique_receipt_ids = set(r[0] for r in session.query(UserPurchases.receipt_id).filter(UserPurchases.user_id == user_id).all())

        items_data, receipt_info = parse_json(receipt_data)
        receipt_id = receipt_info.get('receipt_id', None)
        
        if receipt_id in unique_receipt_ids:
            with get_connection() as session:
                session.query(UserPurchases).filter(UserPurchases.receipt_id == receipt_id).update({'in_history': True})
            await bot.send_message(user_id, text = f"⚠️ {idx_info} Чек с ID {receipt_id} уже добавлен в вашу историю")
            return
        
        data = predict_product_categories(items_data, bert_2level_model, le, min_confidence)

        save_data_for_database(data, receipt_info, user_id)
        
        purchase_date = datetime.strftime(receipt_info['purchase_datetime'], '%d.%m.%Y %H:%M')
        if add_to_history_bool:
            await bot.send_message(user_id, f"✅ {idx_info} Чек от {purchase_date} добавлен в датасет и вашу историю")
        else:
            await bot.send_message(user_id, f"➡️ {idx_info} Чек от {purchase_date} был добавлен в общий датасет, но не был добавлен в вашу историю исходя из ваших настроек")  


    except Exception as e:
        logger.error(f"Error processing receipt {receipt_id if receipt_id else 'unknown'} for user {user_id}: {e}")
        await bot.send_message(user_id, text = f"Произошла ошибка при обработке чека{' с ID ' + receipt_id if receipt_id else ''}")


@dp.message(lambda message: message.document and message.document.file_name.endswith('.json'))
async def handle_json_upload(message: Message):
    """
    Handle a JSON file upload from a user.

    The function creates a temporary file for the JSON data,
    processes it, then automatically cleans up the temporary file.
    """
    user_id = message.from_user.id
        
    try:
        # Create a temporary file that will be automatically cleaned up
        with NamedTemporaryFile(suffix='.json', mode='w+', delete=True) as temp_file:
            # Download the file to the temporary location
            await bot.download(message.document, destination=temp_file.name)
            # Read the JSON data
            temp_file.seek(0)
            data_received = json.load(temp_file)
        
        idx_info = ''
        if isinstance(data_received, list):
            for i, receipt_data in enumerate(data_received, start=1):
                idx_info = f'{i}/{len(data_received)}'
                await process_receipt(receipt_data, user_id, idx_info)
        else:
            await process_receipt(data_received, user_id, idx_info)
        
    except Exception as e:
        await message.answer("Произошла ошибка при обработке файла. Попробуйте еще раз")
        logger.error(f"Error processing file for user {user_id}: {e}")

    # Show the main menu after processing the upload
    back_button = create_back_button(text="Главное меню", callback_data="main_menu")
    await message.answer("Все файлы были обработаны", reply_markup=back_button)
    # Updating user's contribution to dataset
    update_user_contributions(user_id)