from aiogram.types import CallbackQuery
from sqlalchemy import func, and_

from bot.bot_init import bot, dp
from utils.db_utils import (
    get_connection, Users, UserSettings, 
    UserPurchases
)  # Updated imports
from utils.utils import create_back_button

@dp.callback_query(lambda c: c.data == "history_user_profile")
async def handle_user_profile(callback_query: CallbackQuery):
    """Display user profile with statistics."""
    await callback_query.answer()
    await bot.delete_message(
        chat_id=callback_query.message.chat.id, 
        message_id=callback_query.message.message_id
    )
    user_id = callback_query.from_user.id

    # Get all user statistics from database
    stats = await get_user_statistics(user_id)
    
    # Format statistics message
    statistics = format_statistics_message(stats)
    
    # Send formatted statistics
    back_button = create_back_button(
        text="К истории покупок", 
        callback_data="back_to_history_delete"
    )
    await bot.send_message(user_id, statistics, reply_markup=back_button)

async def get_user_statistics(user_id: int) -> dict:
    """Get user statistics from database using SQLAlchemy."""
    with get_connection() as session:
        # Get user data from both users and user_settings tables
        user_data = session.query(
            Users, UserSettings
        ).join(
            UserSettings, Users.user_id == UserSettings.user_id
        ).filter(
            Users.user_id == user_id
        ).first()

        if not user_data:
            return {}

        user, settings = user_data

        # Get total receipts count
        total_receipts = session.query(
            func.count(func.distinct(UserPurchases.receipt_id))
        ).filter(
            UserPurchases.user_id == user_id
        ).scalar() or 0

        # Get history receipts count
        history_receipts = session.query(
            func.count(func.distinct(UserPurchases.receipt_id))
        ).filter(
            and_(
                UserPurchases.user_id == user_id,
                UserPurchases.in_history == True
            )
        ).scalar() or 0

        # Get total spending
        total_spending = session.query(
            func.sum(UserPurchases.total_sum)
        ).filter(
            and_(
                UserPurchases.user_id == user_id,
                UserPurchases.in_history == True
            )
        ).scalar() or 0

        return {
            'user_id': user.user_id,
            'user_name': user.user_name,
            'registration_date': user.registration_date,
            'original_receipts_added': user.original_receipts_added,
            'products_added': user.products_added,
            'add_to_history': settings.add_to_history,
            'minimal_prediction_confidence': settings.minimal_prediction_confidence,
            'total_receipts': total_receipts,
            'history_receipts': history_receipts,
            'total_spending': total_spending
        }

def format_statistics_message(stats: dict) -> str:
    """Format user statistics into a readable message."""
    return (
        f"📊 Общая статистика:\n\n"
        
        f"📝 Всего чеков добавлено: {stats.get('total_receipts', 0)}\n"
        f"📈 Уникальных чеков добавлено: {stats.get('original_receipts_added', 0)}\n"
        f"📈 Чеков в истории: {stats.get('history_receipts', 0)}\n"
        f"🍔 Продуктов добавлено: {stats.get('products_added', 0)}\n"
        f"💸 Всего денег потрачено: {int(stats.get('total_spending', 0))}₽\n\n"

        f"🕰️ Дата регистрации: {stats.get('registration_date', 'Не указана')}\n"
        f"💯 Минимальная уверенность модели: {int(stats.get('minimal_prediction_confidence', 0.5)*100)}%"
    )
