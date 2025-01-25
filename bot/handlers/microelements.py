from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup

from bot.bot_init import bot, dp, MICROELEMENTS_TABLE
from utils.utils import create_back_button
from utils.data_processor import get_microelements_data
from config.config import ABBREVIATION_MICROELEMENTS_DICT

async def display_microelements(user_id, message, microelements_data, page=0):
    """Display paginated list of microelements with their amounts."""
    keyboard = []
    microelements_list = list(microelements_data.items())
    start_index = page * 7
    end_index = min(start_index+7, len(microelements_list))

    # Add microelement buttons
    for i in range(start_index, end_index):
        element, data = microelements_list[i]
        total_amount = data['total_amount']
        unit = 'Ккал' if element == ABBREVIATION_MICROELEMENTS_DICT["эц"] else 'г'
        keyboard.append([
            InlineKeyboardButton(
                text=f"{element}: {total_amount:.2f} {unit}", 
                callback_data=f"display_microelements_element_{i}"
            )
        ])

    # Add navigation buttons
    nav_buttons = []
    if page > 0:
        nav_buttons.append(InlineKeyboardButton(
            text="⬅️ Назад", 
            callback_data=f"display_microelements_page_{page-1}"
        ))
    if end_index < len(microelements_list):
        nav_buttons.append(InlineKeyboardButton(
            text="Вперед ➡️", 
            callback_data=f"display_microelements_page_{page+1}"
        ))
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
    text = "Выберите микроэлемент для просмотра источников:"

    try:
        if message.message_id:
            await bot.edit_message_text(
                chat_id=user_id,
                message_id=message.message_id,
                text=text,
                reply_markup=markup
            )
        else:
            await bot.send_message(
                chat_id=user_id,
                text=text,
                reply_markup=markup
            )
    except Exception as e:
        print(f"Error displaying microelements: {e}")

@dp.callback_query(lambda c: c.data and c.data.startswith("history_microelements"))
async def handle_display_microelements(callback_query: CallbackQuery):
    """Handle microelements display request."""
    await callback_query.answer()
    await bot.delete_message(
        chat_id=callback_query.message.chat.id, 
        message_id=callback_query.message.message_id
    )

    user_id = callback_query.from_user.id
    microelements_data = get_microelements_data(user_id, MICROELEMENTS_TABLE)

    if not microelements_data:
        back_button = create_back_button(
            text="Назад", 
            callback_data="back_to_history_delete"
        )
        await bot.send_message(
            user_id, 
            "Данные о микроэлементах отсутствуют", 
            reply_markup=back_button
        )
        return

    new_message = await bot.send_message(
        user_id, 
        "Загрузка данных о микроэлементах..."
    )
    await display_microelements(user_id, new_message, microelements_data)

@dp.callback_query(lambda c: c.data and c.data.startswith("display_microelements_page_"))
async def handle_microelements_page(callback_query: CallbackQuery):
    """Handle microelements pagination."""
    await callback_query.answer()
    page = int(callback_query.data.split("_")[-1])
    user_id = callback_query.from_user.id
    
    microelements_data = get_microelements_data(user_id, MICROELEMENTS_TABLE)
    await display_microelements(
        user_id, 
        callback_query.message, 
        microelements_data, 
        page
    )

@dp.callback_query(lambda c: c.data and c.data.startswith("display_microelements_element_"))
async def handle_microelement_sources(callback_query: CallbackQuery):
    """Handle individual microelement display."""
    await callback_query.answer()
    element_index = int(callback_query.data.split("_")[-1])
    user_id = callback_query.from_user.id
    
    microelements_data = get_microelements_data(user_id, MICROELEMENTS_TABLE)
    element, data = list(microelements_data.items())[element_index]
    
    # Format sources text
    sources_text = format_microelement_sources(element, data)
    
    # Send sources information
    back_button = InlineKeyboardMarkup(
        inline_keyboard=[[
            InlineKeyboardButton(
                text="Назад к микроэлементам", 
                callback_data="history_microelements"
            )
        ]]
    )
    
    await send_microelement_sources(user_id, sources_text, back_button, callback_query.message)

def format_microelement_sources(element: str, data: dict) -> str:
    """Format microelement sources into displayable text."""
    unit = 'Ккал' if element == ABBREVIATION_MICROELEMENTS_DICT["эц"] else 'г'
    
    sources_text = (
        f"Общее количество: {data['total_amount']:.2f} {unit}\n"
        f"Источники {element}:\n\n"
    )
    
    sources_text += '\n'.join(
        f"{i}. {source}: {amount:.6f} {unit}"
        for i, (source, amount) in enumerate(data['sources'], start=1)
    )
    
    return sources_text

async def send_microelement_sources(
    user_id: int, 
    sources_text: str, 
    back_button: InlineKeyboardMarkup, 
    original_message
):
    """Send microelement sources, splitting into chunks if needed."""
    try:
        if len(sources_text) < 4096:
            await bot.edit_message_text(
                chat_id=user_id,
                message_id=original_message.message_id,
                text=sources_text,
                reply_markup=back_button
            )
        else:
            chunks = split_message_into_chunks(sources_text)
            for i, chunk in enumerate(chunks):
                if i == 0:
                    await bot.edit_message_text(
                        chat_id=user_id,
                        message_id=original_message.message_id,
                        text=chunk
                    )
                elif i == len(chunks) - 1:
                    await bot.send_message(
                        chat_id=user_id,
                        text=chunk,
                        reply_markup=back_button
                    )
                else:
                    await bot.send_message(
                        chat_id=user_id,
                        text=chunk
                    )
    except Exception as e:
        print(f"Error sending microelement sources: {e}")

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
        chunks.append(current_chunk.rstrip())
    
    return chunks
