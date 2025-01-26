from aiogram.types import CallbackQuery

from bot.bot_init import bot, dp
from utils.db_utils import db_connection
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
    """Get user statistics from database."""
    with db_connection() as conn:        
        cursor = conn.cursor()
        # Get user data from both users and user_settings tables
        cursor.execute("""
            SELECT u.user_id, u.user_name, u.registration_date, u.original_receipts_added,
                   u.products_added, us.add_to_history, us.minimal_prediction_confidence
            FROM users u
            JOIN user_settings us ON u.user_id = us.user_id
            WHERE u.user_id = ?
        """, (user_id,))
        user_data = dict(cursor.fetchone() or {})

        # Get receipt counts
        cursor.execute("""
            SELECT COUNT(DISTINCT receipt_id) 
            FROM user_purchases
            WHERE user_id = ?
        """, (user_id,))
        user_data['total_receipts'] = cursor.fetchone()['COUNT(DISTINCT receipt_id)']

        cursor.execute("""
            SELECT COUNT(DISTINCT receipt_id) 
            FROM user_purchases
            WHERE user_id = ? AND in_history = 1
        """, (user_id,))
        user_data['history_receipts'] = cursor.fetchone()['COUNT(DISTINCT receipt_id)']

        # Get total spending
        cursor.execute("""
            SELECT SUM(total_sum) 
            FROM user_purchases
            WHERE user_id = ? AND in_history = 1
        """, (user_id,))
        user_data['total_spending'] = cursor.fetchone()['SUM(total_sum)'] or 0

    return user_data

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
