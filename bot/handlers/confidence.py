from aiogram import F
from aiogram.types import Message, CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup

from bot.bot_init import bot, dp
from utils.db_utils import db_connection
from utils.utils import create_back_button
from bot.handlers.menu_utils import show_menu_to_user  # Change this line

# Store message IDs for cleanup
set_confidence_markup_queue = {}

@dp.message(F.text.startswith('/confidence'))
async def set_confidence(message: Message, user_id=None):
    """
    Ask user to input a new confidence threshold.
    
    Sends explanation message and input prompt with option to skip.
    Stores message IDs for later cleanup.
    """
    user_id = user_id if user_id else message.from_user.id 
    
    # Send explanation message
    first_message = await message.answer(
        'Вы можете изменить минимальную уверенность модели, отправив число от 1 до 99 (в процентах).\n\n'
        'Например, "35" установит минимальную уверенность на 35%. '
        'Предсказания с меньшей уверенностью будут считаться нераспознанными.\n'
        'Рекомендуем выбирать значение от 20% до 85%.\n'
        'Подробнее читайте в /help.',
        parse_mode='Markdown'
    )

    # Create skip button
    skip_button = InlineKeyboardMarkup(
        inline_keyboard=[[
            InlineKeyboardButton(
                text="Пропустить изменение уверенности модели",
                callback_data=f"skip_confidence_change_{user_id}"
            )
        ]]
    )

    # Send input prompt
    second_message = await message.answer(
        'Отправьте целое число от 1 до 100 своим следующим сообщением:', 
        reply_markup=skip_button
    )
    
    # Store message IDs for cleanup
    set_confidence_markup_queue[user_id] = {
        'first_message': first_message.message_id,
        'second_message': second_message.message_id
    }

@dp.callback_query(lambda c: c.data and c.data.startswith("skip_confidence_change_"))
async def handle_skip_confidence_change(callback_query: CallbackQuery):
    """Handle skipping confidence threshold change."""
    user_id = callback_query.from_user.id
    await callback_query.answer()

    # Clean up previous messages
    await cleanup_confidence_messages(user_id)

    # Inform user and return to menu
    await callback_query.message.answer(
        "Изменение минимальной уверенности было пропущено. "
        "Вы можете изменить её позже через команду /confidence."
    )
    await show_menu_to_user(callback_query.message, user_id)

@dp.message(lambda message: message.text and message.text.isdigit() and 1 <= int(message.text) <= 99)
async def handle_set_confidence(message: Message, user_id=None):
    """
    Handle confidence value input from user.
    
    Updates confidence threshold in database if valid value provided.
    """
    user_id = user_id if user_id else message.from_user.id 
    display_confidence_value = float(message.text)
    confidence_value = display_confidence_value / 100

    # Clean up previous messages
    await cleanup_confidence_messages(user_id)

    # Update confidence in database
    await update_confidence_threshold(user_id, confidence_value)

    # Send confirmation
    back_button = create_back_button(text="Назад", callback_data="main_menu")
    await message.answer(
        f"Минимальная уверенность модели установлена на {display_confidence_value}%.",
        reply_markup=back_button
    )

async def cleanup_confidence_messages(user_id: int):
    """Clean up stored confidence setting messages."""
    message_ids = set_confidence_markup_queue.get(user_id, {})
    for msg_id in message_ids.values():
        try:
            await bot.delete_message(chat_id=user_id, message_id=msg_id)
        except Exception:
            pass  # Message might have been deleted already
    set_confidence_markup_queue.pop(user_id, None)

async def update_confidence_threshold(user_id: int, confidence_value: float):
    """Update user's confidence threshold in database."""
    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE user_settings SET minimal_prediction_confidence = ? WHERE user_id = ?",
            (confidence_value, user_id)
        )
        conn.commit()
