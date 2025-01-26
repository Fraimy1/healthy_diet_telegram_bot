"""Centralized menu utilities to avoid circular imports."""
from aiogram.types import Message, InlineKeyboardButton, InlineKeyboardMarkup
from utils.db_utils import db_connection

async def show_menu_to_user(message: Message, user_id: int = None):
    """Show menu to user."""
    from bot.handlers.menu import show_main_menu
    await show_main_menu(message, user_id)

async def get_user_settings(user_id: int) -> dict:
    """Get user settings from database."""
    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT minimal_prediction_confidence, add_to_history, return_excel_document 
            FROM user_settings 
            WHERE user_id = ?""", (user_id,))
        user_data = cursor.fetchone()
        return dict(user_data) if user_data else {}

def create_menu_keyboard(user_data: dict) -> InlineKeyboardMarkup:
    """Create menu keyboard with current user settings."""
    min_confidence = int(user_data.get('minimal_prediction_confidence', 0.5) * 100)
    add_to_history_emoji = '✅' if user_data.get('add_to_history', False) else '❌'
    
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="📖 Инструкция", callback_data="menu_instruction"),
                InlineKeyboardButton(text="❓ Помощь", callback_data="menu_help")
            ],
            [
                InlineKeyboardButton(
                    text=f"⚙️ Изменить уверенность - {min_confidence}%", 
                    callback_data="menu_confidence"
                ),
            ],
            [
                InlineKeyboardButton(
                    text=f"📝 Добавлять в историю {add_to_history_emoji}", 
                    callback_data="menu_add_to_history"
                ),
            ],
            [            
                InlineKeyboardButton(text="📜 История покупок", callback_data="menu_history"),
            ]
        ]
    )
