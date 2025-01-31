from aiogram import F
from aiogram.types import Message, CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup
import json
from tempfile import NamedTemporaryFile
from datetime import datetime

from bot.bot_init import bot, dp, bert_2level_model, le
from utils.db_utils import get_connection, UserSettings, UserPurchases  # Updated import
from utils.utils import create_back_button, update_user_contributions
from bot.handlers.menu_utils import show_menu_to_user
from utils.logger import logger
from utils.data_processor import parse_json, predict_product_categories, save_data_for_database
from config.config import SHOW_PROGRESS_TIMES  # New import for progress update frequency

# Global in‐memory storage for receipt details per user.
# Structure: { user_id: { "added_to_user_history": [receipt_record, ...],
#                           "added_to_dataset_only": [...],
#                           "already_added": [...],
#                           "error": [...] } }
pending_receipts = {}

# Global storage for the last message(s) sent to each user.
# This is used to “delete” the previous message(s) before sending a new one.
last_messages = {}  # { user_id: [message_id1, message_id2, ...] }


##########################################
# Helpers for message sending and splitting
##########################################

async def safe_send_message(user_id: int, text: str, reply_markup: InlineKeyboardMarkup = None):
    """
    Deletes the previous message(s) for the user (if any) and sends a new message.
    """
    if user_id in last_messages:
        for mid in last_messages[user_id]:
            try:
                await bot.delete_message(user_id, mid)
            except Exception as e:
                logger.error(f"Error deleting previous message {mid} for user {user_id}: {e}")
    sent_message = await bot.send_message(user_id, text, reply_markup=reply_markup)
    last_messages[user_id] = [sent_message.message_id]


async def safe_send_messages(user_id: int, texts: list, reply_markup: InlineKeyboardMarkup = None):
    """
    Sends a list of text messages (for when the text must be split) after deleting previous ones.
    Only the last message will receive the reply_markup.
    """
    if user_id in last_messages:
        for mid in last_messages[user_id]:
            try:
                await bot.delete_message(user_id, mid)
            except Exception as e:
                logger.error(f"Error deleting previous message {mid} for user {user_id}: {e}")
    last_messages[user_id] = []
    for i, text in enumerate(texts):
        rm = reply_markup if i == len(texts) - 1 else None
        sent = await bot.send_message(user_id, text, reply_markup=rm)
        last_messages[user_id].append(sent.message_id)


def split_long_message(text: str, max_length: int = 4000) -> list:
    """
    Splits a long message into a list of messages each with at most max_length characters.
    It first tries to split on newline boundaries.
    """
    if len(text) <= max_length:
        return [text]
    lines = text.splitlines(keepends=True)
    messages = []
    current_message = ""
    for line in lines:
        if len(current_message) + len(line) > max_length:
            messages.append(current_message)
            current_message = line
        else:
            current_message += line
    if current_message:
        messages.append(current_message)
    # In case any split piece is still too long, do a hard split.
    final_messages = []
    for msg in messages:
        if len(msg) > max_length:
            final_messages.extend([msg[i:i+max_length] for i in range(0, len(msg), max_length)])
        else:
            final_messages.append(msg)
    return final_messages


def get_report_options_keyboard() -> InlineKeyboardMarkup:
    """
    Returns the inline keyboard attached to the main processing report.
    """
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="✅ Добавлено в историю", callback_data="show_added_to_user_history"),
            InlineKeyboardButton(text="✅ Добавлено в датасет", callback_data="show_added_to_dataset_only")
        ],
        [
            InlineKeyboardButton(text="⚠️ Уже добавлено", callback_data="show_already_added"),
            InlineKeyboardButton(text="❌ Ошибки", callback_data="show_error")
        ],
        [
            InlineKeyboardButton(text="📖 Меню", callback_data="show_menu")
        ]
    ])


def get_report_options_text(user_id: int) -> str:
    """
    Creates a summary report based on the details stored in pending_receipts.
    """
    user_data = pending_receipts.get(user_id, {})
    added_to_user_history = len(user_data.get('added_to_user_history', []))
    added_to_dataset_only = len(user_data.get('added_to_dataset_only', []))
    already_added = len(user_data.get('already_added', []))
    error = len(user_data.get('error', []))
    report = "Результаты обработки файлов:\n"
    report += f"✅ Добавлено в вашу историю: {added_to_user_history}\n"
    report += f"✅ Добавлено только в датасет, исходя из ваших настроек: {added_to_dataset_only}\n"
    report += f"⚠️ Уже добавлено в вашу историю: {already_added}\n"
    report += f"❌ Ошибок при обработке: {error}\n"
    report += "Для получения ID и даты добавленных чеков используйте кнопки снизу"
    return report


##########################################
# Receipt Processing
##########################################

async def process_receipt(receipt_data, user_id, idx_info=""):
    """
    Processes a receipt and updates the database accordingly.
    Now it extracts both receipt_id and purchase_date.
    Returns a tuple (result_status, receipt_record), where receipt_record is a dict.
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
        purchase_date = receipt_info.get('purchase_date', None)
        if not purchase_date:
            purchase_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        receipt_record = {"id": receipt_id, "purchase_date": purchase_date}

        if receipt_id in unique_receipt_ids:
            with get_connection() as session:
                session.query(UserPurchases).filter(UserPurchases.receipt_id == receipt_id).update({'in_history': True})
            return 'already_added', receipt_record

        data = predict_product_categories(items_data, bert_2level_model, le, min_confidence)
        save_data_for_database(data, receipt_info, user_id)

        if add_to_history_bool:
            return 'added_to_user_history', receipt_record
        else:
            return 'added_to_dataset_only', receipt_record

    except Exception as e:
        logger.error(f"Error processing receipt {receipt_id if receipt_id else 'unknown'} for user {user_id}: {e}")
        return 'error', {"id": receipt_id, "purchase_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


@dp.message(lambda message: message.document and message.document.file_name.endswith('.json'))
async def handle_json_upload(message: Message):
    """
    Handles a JSON file upload from a user.
    Creates a temporary file, processes its receipts (while updating a progress bar), 
    and then sends the main processing report.
    """
    user_id = message.from_user.id

    try:
        with NamedTemporaryFile(suffix='.json', mode='w+', delete=True) as temp_file:
            await bot.download(message.document, destination=temp_file.name)
            temp_file.seek(0)
            data_received = json.load(temp_file)

        idx_info = ''
        # Dictionaries to count results and store detailed receipt records.
        results = {'added_to_user_history': 0, 'added_to_dataset_only': 0, 'already_added': 0, 'error': 0}
        result_ids = {'added_to_user_history': [], 'added_to_dataset_only': [], 'already_added': [], 'error': []}

        # If we are processing a list of receipts, show a progress bar
        if isinstance(data_received, list):
            total = len(data_received)
            progress_interval = max(1, total // SHOW_PROGRESS_TIMES)
            # Send the initial progress message.
            progress_msg = await bot.send_message(user_id, f"0/{total} чеков было обработано")
            for i, receipt_data in enumerate(data_received, start=1):
                idx_info = f'{i}/{total}'
                result, receipt_record = await process_receipt(receipt_data, user_id, idx_info)
                results[result] += 1
                result_ids[result].append(receipt_record)

                # Update progress message periodically.
                if i % progress_interval == 0 or i == total:
                    try:
                        await bot.edit_message_text(
                            f"{i}/{total} чеков было обработано",
                            chat_id=user_id,
                            message_id=progress_msg.message_id
                        )
                    except Exception as e:
                        logger.error(f"Error updating progress message for user {user_id}: {e}")
            # Once processing is complete, remove the progress message.
            try:
                await bot.delete_message(user_id, progress_msg.message_id)
            except Exception as e:
                logger.error(f"Error deleting progress message for user {user_id}: {e}")
        else:
            result, receipt_record = await process_receipt(data_received, user_id, idx_info)
            results[result] += 1
            result_ids[result].append(receipt_record)

        # Save detailed receipt data for later retrieval.
        pending_receipts[user_id] = result_ids

        # Build and send the main report.
        report = get_report_options_text(user_id)
        update_user_contributions(user_id)
        await safe_send_message(user_id, report, reply_markup=get_report_options_keyboard())

    except Exception as e:
        back_button = create_back_button()
        await message.answer("Произошла ошибка при обработке файла. Попробуйте еще раз", reply_markup=back_button)
        logger.error(f"Error processing file for user {user_id}: {e}")


##########################################
# Callback Query Handlers for Report Details
##########################################

@dp.callback_query(lambda c: c.data in [
    "show_added_to_user_history",
    "show_added_to_dataset_only",
    "show_already_added",
    "show_error"
])
async def handle_display_options(callback_query: CallbackQuery):
    """
    Displays detailed statistics (receipt IDs and purchase dates) for the selected category.
    If the message is too long, it is split into multiple messages.
    A "back" button is provided to return to the main report.
    """
    user_id = callback_query.from_user.id
    # Extract the key by removing the "show_" prefix.
    key = callback_query.data[len("show_"):]
    receipt_records = pending_receipts.get(user_id, {}).get(key, [])
    if not receipt_records:
        text = "Нет данных для отображения"
    else:
        lines = []
        for record in receipt_records:
            lines.append(f"Чек ID: {record.get('id', 'N/A')}, Дата покупки: {record.get('purchase_date', 'N/A')}")
        text = "\n".join(lines)

    messages = split_long_message(text)
    back_kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="⬅️ Назад", callback_data="back_to_report")]
    ])
    await safe_send_messages(user_id, messages, reply_markup=back_kb)
    await callback_query.answer()


@dp.callback_query(lambda c: c.data == "back_to_report")
async def back_to_report(callback_query: CallbackQuery):
    """
    Returns the user back to the main processing report menu.
    """
    user_id = callback_query.from_user.id
    report = get_report_options_text(user_id)
    await safe_send_message(user_id, report, reply_markup=get_report_options_keyboard())
    await callback_query.answer()


@dp.callback_query(lambda c: c.data == "show_menu")
async def go_to_menu(callback_query: CallbackQuery):
    """
    Sends the user back to the main menu.
    Before sending the new menu, any previous messages are deleted.
    """
    user_id = callback_query.from_user.id
    if user_id in last_messages:
        for mid in last_messages[user_id]:
            try:
                await bot.delete_message(user_id, mid)
            except Exception as e:
                logger.error(f"Error deleting previous message {mid} for user {user_id}: {e}")
        last_messages[user_id] = []
    await show_menu_to_user(callback_query.message, user_id)
    await callback_query.answer()
