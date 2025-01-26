from aiogram import F
from aiogram.types import Message, CallbackQuery

from bot.bot_init import bot, dp
from bot.handlers.menu_utils import get_user_settings, create_menu_keyboard
from utils.logger import logger

@dp.message(F.text.startswith('/show_menu'))
async def show_main_menu(message: Message, user_id=None):
    """Display main menu with current user settings."""
    user_id = user_id if user_id else message.from_user.id
    logger.info(f"Showing main menu to user {user_id}")
    
    try:
        await bot.delete_message(chat_id=message.chat.id, message_id=message.message_id)
    except Exception as e:
        logger.debug(f"Could not delete message for user {user_id}: {e}")

    user_data = await get_user_settings(user_id)
    logger.debug(f"Retrieved settings for user {user_id}: {user_data}")
    keyboard = create_menu_keyboard(user_data)
    await message.answer("Меню", reply_markup=keyboard)

@dp.callback_query(lambda c: c.data and c.data.startswith("menu_"))
async def handle_menu_callbacks(callback_query: CallbackQuery):
    """Handle all menu-related callback queries."""
    data = callback_query.data
    user_id = callback_query.from_user.id
    logger.info(f"Menu callback from user {user_id}: {data}")
    await callback_query.answer()

    try:
        await bot.delete_message(
            chat_id=callback_query.message.chat.id, 
            message_id=callback_query.message.message_id
        )
    except:
        pass
    
    user_id = callback_query.from_user.id
    
    # Import handlers only when needed to avoid circular imports
    if data == "menu_instruction":
        from bot.handlers.instruction import show_instruction
        await show_instruction(callback_query.message, user_id)
    elif data == "menu_help":
        from bot.handlers.help import help_command
        await help_command(callback_query.message)
    elif data == "menu_confidence":
        from bot.handlers.confidence import set_confidence
        await set_confidence(callback_query.message, user_id)
    elif data == "menu_history":
        from bot.handlers.history import show_history_options
        await show_history_options(callback_query.message)
    elif data == "menu_add_to_history":
        from bot.handlers.history import add_to_history
        await add_to_history(callback_query.message, user_id)
