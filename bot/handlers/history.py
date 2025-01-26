from collections import defaultdict
from datetime import datetime
from aiogram import F
from aiogram.types import Message, CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup

from bot.bot_init import bot, dp
from utils.db_utils import db_connection
from utils.utils import create_back_button
from utils.data_processor import get_sorted_user_receipts, count_product_amounts
from bot.handlers.menu_utils import show_menu_to_user  # Change this line
from utils.logger import logger

# Track messages for categories display
user_messages_categories = defaultdict(list)

@dp.message(F.text.startswith('/show_history'))
async def show_history_options(message: Message):
    """Display main history menu with options."""
    user_id = message.from_user.id
    logger.info(f"User {user_id} accessed history menu")
    try:
        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(text="🧾 Чеки", callback_data="history_receipts"),
                    InlineKeyboardButton(text="📊 Профиль", callback_data="history_user_profile"),
                ],
                [
                    InlineKeyboardButton(text="🍎 Категории продуктов", callback_data="history_categories"),
                ],
                [
                    InlineKeyboardButton(text="💊 Микроэлементы", callback_data="history_microelements"),
                ],
                [
                    InlineKeyboardButton(text="🚪 Выйти", callback_data="history_main_menu"),
                ]
            ]
        )  
        await message.answer('📜 История покупок', reply_markup=keyboard)
        logger.debug(f"Successfully displayed history menu to user {user_id}")
    except Exception as e:
        logger.error(f"Failed to show history menu to user {user_id}: {str(e)}", exc_info=True)
        raise

@dp.callback_query(lambda c: c.data and c.data.endswith("main_menu"))
async def handle_history_quit(callback_query: CallbackQuery):
    """Handle exit from history menu."""
    user_id = callback_query.from_user.id
    await callback_query.answer()
    await bot.delete_message(
        chat_id=callback_query.message.chat.id, 
        message_id=callback_query.message.message_id
    )
    await show_menu_to_user(callback_query.message, user_id)

@dp.callback_query(lambda c: c.data and c.data.startswith("back_to_history"))
async def handle_back_to_history(callback_query: CallbackQuery):
    """Handle back navigation to history menu."""
    await callback_query.answer()
    await callback_query.message.edit_reply_markup(reply_markup=None)
    if callback_query.data.endswith("delete"):
        await bot.delete_message(
            chat_id=callback_query.message.chat.id, 
            message_id=callback_query.message.message_id
        )
    await show_history_options(callback_query.message)

@dp.message(F.text.startswith('/add_to_history'))    
async def add_to_history(message: Message, user_id=None):
    """Toggle add_to_history setting for user."""
    user_id = user_id if user_id else message.from_user.id 

    # Toggle setting in database
    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT add_to_history FROM user_settings WHERE user_id = ?", 
            (user_id,)
        )
        add_to_history_value = not cursor.fetchone()[0]
        cursor.execute(
            "UPDATE user_settings SET add_to_history = ? WHERE user_id = ?", 
            (add_to_history_value, user_id)
        )
        conn.commit()

    # Send confirmation message
    back_button = create_back_button(text="Главное меню", callback_data="main_menu")
    status = "включено" if add_to_history_value else "отключено"
    emoji = "✅" if add_to_history_value else "❌"
    await message.answer(
        f"{emoji} Добавление чеков в вашу историю {status}.", 
        reply_markup=back_button
    )

# Receipt display functions
async def display_receipts(user_id, message, receipts, page=0):
    """Display paginated list of receipts."""
    logger.debug(f"Displaying receipts page {page} for user {user_id}")
    try:
        keyboard = []
        start_index = page * 7
        end_index = min(start_index + 7, len(receipts))

        # Add receipt buttons
        for i in range(start_index, end_index):
            receipt = receipts[i]
            date = datetime.strptime(receipt['purchase_datetime'], '%Y-%m-%d %H:%M:%S')  # Fixed format specifier
            display_date = date.strftime('%d.%m.%Y %H:%M')
            keyboard.append([
                InlineKeyboardButton(
                    text=f"Чек от {display_date}", 
                    callback_data=f"display_receipts_receipt_{receipt['receipt_id']}"
                )
            ])

        # Add navigation buttons
        nav_buttons = []
        if page > 0:
            nav_buttons.append(InlineKeyboardButton(
                text="⬅️ Назад", 
                callback_data=f"display_receipts_page_{page-1}"
            ))
        if end_index < len(receipts):
            nav_buttons.append(InlineKeyboardButton(
                text="Вперед ➡️", 
                callback_data=f"display_receipts_page_{page+1}"
            ))
        if nav_buttons:
            keyboard.append(nav_buttons)

        # Add back button
        keyboard.append([InlineKeyboardButton(
            text="К истории покупок", 
            callback_data="back_to_history_delete"
        )])

        markup = InlineKeyboardMarkup(inline_keyboard=keyboard)

        # Send or edit message
        if message.message_id:
            await bot.edit_message_text(
                chat_id=user_id,
                message_id=message.message_id,
                text="Выберите чек для просмотра:",
                reply_markup=markup
            )
        else:
            await bot.send_message(
                chat_id=user_id,
                text="Выберите чек для просмотра:",
                reply_markup=markup
            )

        logger.info(f"Successfully displayed {end_index - start_index} receipts to user {user_id}")
    except Exception as e:
        logger.error(f"Failed to display receipts for user {user_id}: {str(e)}", exc_info=True)
        raise

    return markup

@dp.callback_query(lambda c: c.data == "history_receipts")
async def handle_history_receipts(callback_query: CallbackQuery):
    user_id = callback_query.from_user.id
    logger.info(f"User {user_id} accessed receipt history")
    
    await callback_query.answer()
    await bot.delete_message(
        chat_id=callback_query.message.chat.id, 
        message_id=callback_query.message.message_id
    )

    user_id = callback_query.from_user.id
    sorted_receipts = get_sorted_user_receipts(user_id)

    if not sorted_receipts:
        back_button = create_back_button(text="Назад", callback_data="back_to_history_delete")
        await bot.send_message(
            user_id, 
            "Ваша история покупок пока пуста", 
            reply_markup=back_button
        )
        return

    new_message = await bot.send_message(user_id, "Загрузка истории покупок...")
    await display_receipts(user_id, new_message, sorted_receipts)

@dp.callback_query(lambda c: c.data and c.data.startswith("display_receipts"))
async def handle_display_receipts(callback_query: CallbackQuery):
    """Handle receipt display callbacks."""
    await callback_query.answer()
    data = callback_query.data
    user_id = callback_query.from_user.id

    if data.startswith("display_receipts_page_"):
        page = int(data[len("display_receipts_page_"):])
        sorted_receipts = get_sorted_user_receipts(user_id)
        await display_receipts(user_id, callback_query.message, sorted_receipts, page)
    
    elif data.startswith("display_receipts_receipt_"):
        await bot.delete_message(
            chat_id=callback_query.message.chat.id, 
            message_id=callback_query.message.message_id
        )
        receipt_id = data[len("display_receipts_receipt_"):].strip()
        await display_single_receipt(user_id, receipt_id)

async def display_single_receipt(user_id: int, receipt_id: str):
    """Display details of a single receipt."""
    with db_connection() as conn:
        cursor = conn.cursor()
        # Get receipt items
        cursor.execute("""
            SELECT *
            FROM receipt_items
            WHERE receipt_id = ?
        """, (receipt_id,))
        receipt_items = cursor.fetchall()

        # Get receipt info
        cursor.execute("""
            SELECT *
            FROM user_purchases
            WHERE receipt_id = ?
        """, (receipt_id,))
        receipt = cursor.fetchone()

    # Format receipt items
    items = [
        f'{i}. "{item["product_name"]}" - {item["quantity"]} - '
        f'{item["user_prediction"]} ({int(item["confidence"]*100)})%'
        for i, item in enumerate(receipt_items, start=1)
    ]

    # Create message
    message = (
        f'{receipt["purchase_datetime"]} - {receipt["total_sum"]}₽\n'
        'Продукты:\n\n' + 
        '\n'.join(items)
    )

    back_button = create_back_button(text="Назад", callback_data="history_receipts")
    await bot.send_message(user_id, message, reply_markup=back_button)
