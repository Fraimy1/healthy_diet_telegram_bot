from collections import defaultdict
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup

from bot.main import logger
from bot.bot_init import bot, dp
from utils.utils import create_back_button
from utils.data_processor import count_product_amounts
from utils.logger import logger

# Track messages for cleanup
user_messages_categories = defaultdict(list)

async def display_categories(user_id, message, product_counts, undetected_categories, page=0):
    """
    Display paginated list of categories with their amounts.
    
    Args:
        user_id: User's Telegram ID
        message: Original message to edit
        product_counts: Dictionary of categorized products with amounts
        undetected_categories: Dictionary of uncategorized products
        page: Current page number for pagination
    """
    logger.debug(
        f"Displaying categories for user {user_id}, page {page}\n"
        f"Product counts: {len(product_counts)}, Undetected: {len(undetected_categories)}"
    )
    
    keyboard = []
    categories_list = list(product_counts.items()) + list(undetected_categories.items())
    start_index = page * 7
    end_index = min(start_index + 7, len(categories_list))

    # Add category buttons
    for i in range(start_index, end_index):
        category, data = categories_list[i]
        total_amount = data['total_amount']
        if isinstance(total_amount, str):
            logger.info
        display_text = (f"{category}: не определено" if total_amount is None 
                       else f"{category}: {total_amount:.2f} г")
        keyboard.append([
            InlineKeyboardButton(
                text=display_text, 
                callback_data=f"display_categories_category_{i}"
            )
        ])

    # Add navigation buttons
    nav_buttons = []
    if page > 0:
        nav_buttons.append(
            InlineKeyboardButton(
                text="⬅️ Назад", 
                callback_data=f"display_categories_page_{page-1}"
            )
        )
    if end_index < len(categories_list):
        nav_buttons.append(
            InlineKeyboardButton(
                text="Вперед ➡️", 
                callback_data=f"display_categories_page_{page+1}"
            )
        )
    if nav_buttons:
        keyboard.append(nav_buttons)

    # Add back button
    keyboard.append([
        InlineKeyboardButton(
            text="К истории покупок", 
            callback_data="back_to_history_delete"
        )
    ])

    markup = InlineKeyboardMarkup(inline_keyboard=keyboard)
    
    # Update or send message
    try:
        if message.message_id:
            await bot.edit_message_text(
                chat_id=user_id,
                message_id=message.message_id,
                text="Выберите категорию для просмотра источников:",
                reply_markup=markup
            )
        else:
            await bot.send_message(
                chat_id=user_id,
                text="Выберите категорию для просмотра источников:",
                reply_markup=markup
            )
        logger.info(f"Successfully displayed categories page {page} to user {user_id}")
    except Exception as e:
        logger.error(f"Failed to display categories for user {user_id}: {e}")
        logger.debug(f"Error details: {str(e)}")
        raise

@dp.callback_query(lambda c: c.data == "history_categories")
async def handle_display_categories(callback_query: CallbackQuery):
    """Handle categories display request."""
    await callback_query.answer()
    user_id = callback_query.from_user.id
    
    # Clean up previous message
    await bot.delete_message(
        chat_id=callback_query.message.chat.id,
        message_id=callback_query.message.message_id
    )

    # Clean up any previous category messages
    await cleanup_category_messages(user_id)

    # Get category data
    product_counts, undetected_categories = count_product_amounts(user_id)

    if not product_counts and not undetected_categories:
        back_button = create_back_button(
            text="К истории покупок", 
            callback_data="back_to_history_delete"
        )
        sent_message = await bot.send_message(
            user_id, 
            "Ваша история покупок пока пуста", 
            reply_markup=back_button
        )
        user_messages_categories[user_id] = [sent_message.message_id]
        return

    # Show categories
    new_message = await bot.send_message(
        user_id, 
        "Загрузка данных о категориях продуктов..."
    )
    user_messages_categories[user_id] = [new_message.message_id]
    await display_categories(
        user_id, 
        new_message, 
        product_counts, 
        undetected_categories
    )

@dp.callback_query(lambda c: c.data and c.data.startswith("display_categories_page_"))
async def handle_categories_page(callback_query: CallbackQuery):
    """Handle category pagination."""
    await callback_query.answer()
    page = int(callback_query.data.split("_")[-1])
    user_id = callback_query.from_user.id
    
    product_counts, undetected_categories = count_product_amounts(user_id)
    await display_categories(
        user_id, 
        callback_query.message, 
        product_counts, 
        undetected_categories, 
        page
    )

@dp.callback_query(lambda c: c.data and c.data.startswith("display_categories_category_"))
async def handle_category_sources(callback_query: CallbackQuery):
    """Handle individual category display."""
    await callback_query.answer()
    category_index = int(callback_query.data.split("_")[-1])
    user_id = callback_query.from_user.id
    
    # Clean up previous messages
    await cleanup_category_messages(user_id)

    # Get category data
    product_counts, undetected_categories = count_product_amounts(user_id)
    categories_list = list(product_counts.items()) + list(undetected_categories.items())
    category, data = categories_list[category_index]
    
    # Format sources text
    sources_text = format_category_sources(category, data)
    
    # Create back button
    back_button = InlineKeyboardMarkup(
        inline_keyboard=[[
            InlineKeyboardButton(
                text="Назад к категориям", 
                callback_data="history_categories"
            )
        ]]
    )
    
    # Send message(s)
    await send_category_sources(user_id, sources_text, back_button)

async def cleanup_category_messages(user_id: int):
    """Clean up stored category messages."""
    for message_id in user_messages_categories[user_id]:
        try:
            await bot.delete_message(chat_id=user_id, message_id=message_id)
        except Exception:
            pass  # Message might have been deleted already
    user_messages_categories[user_id].clear()

def format_category_sources(category: str, data: dict) -> str:
    """Format category sources into displayable text."""
    if data['total_amount'] is None:
        sources_text = f"Категория: {category}\n\nНеопределенные продукты:\n\n"
        sources_text += "\n".join(
            f"{i}. {source}" 
            for i, source in enumerate(data['sources'], start=1)
        )
    else:
        sources_text = (
            f"Категория: {category}\n"
            f"Общее количество: {data['total_amount']:.2f} г\n\n"
            "Источники:\n\n"
        )
        sources_text += "\n".join(
            f"{i}. {source}: {amount:.2f} г" 
            for i, (source, amount) in enumerate(data['sources'], start=1)
        )
    return sources_text

async def send_category_sources(user_id: int, sources_text: str, back_button: InlineKeyboardMarkup):
    """Send category sources, splitting into chunks if needed."""
    if len(sources_text) < 4096:
        sent_message = await bot.send_message(
            chat_id=user_id,
            text=sources_text,
            reply_markup=back_button
        )
        user_messages_categories[user_id] = [sent_message.message_id]
    else:
        chunks = split_message_into_chunks(sources_text)
        for i, chunk in enumerate(chunks):
            reply_markup = back_button if i == len(chunks) - 1 else None
            sent_message = await bot.send_message(
                chat_id=user_id,
                text=chunk,
                reply_markup=reply_markup
            )
            user_messages_categories[user_id].append(sent_message.message_id)

def split_message_into_chunks(text: str, chunk_size: int = 4096) -> list:
    """Split long messages into Telegram-friendly chunks."""
    chunks = []
    current_chunk = ""
    
    for line in text.split("\n"):
        if len(current_chunk) + len(line) + 1 > chunk_size:
            chunks.append(current_chunk)
            current_chunk = line
        else:
            current_chunk += line + "\n"
            
    if current_chunk:
        chunks.append(current_chunk[:-1])  # Remove trailing newline
        
    return chunks
