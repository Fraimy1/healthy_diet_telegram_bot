from aiogram import F
from aiogram.types import Message

from bot.bot_init import dp
from utils.utils import create_back_button
from config.config import HELP_TEXT

@dp.message(F.text.startswith('/help'))
async def help_command(message: Message):
    """
    Handle the /help command.
    
    Displays comprehensive help information about bot usage, including:
    - Basic commands
    - Instructions for receipt processing
    - Information about confidence thresholds
    - History and statistics features
    
    Args:
        message (Message): The message containing the /help command
    """
    back_button = create_back_button(text="Главное меню", callback_data="main_menu")
    await message.answer(
        text=HELP_TEXT,
        parse_mode='Markdown',
        reply_markup=back_button
    )

# Export for use in other modules
async def show_help_to_user(message: Message):
    """Utility function to show help from other modules."""
    await help_command(message)
