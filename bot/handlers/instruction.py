import os
import asyncio
from aiogram import F
from aiogram.types import Message, CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, FSInputFile

from bot.bot_init import bot, dp
from config.config import INSTRUCTION_IMAGES_PATH, DOWNLOADING_INSTRUCTIONS, USAGE_INSTRUCTIONS
from bot.handlers.menu_utils import show_menu_to_user  # Change this line
from bot.handlers.start import get_active_users

# Track users waiting for next button press
waiting_users = {}

async def show_instruction_step(user_id, text, image_path=None, parse_mode='Markdown', button_text='Далее'):
    """Send an instruction step with optional image and wait for user confirmation."""
    callback_data = f"next_{user_id}"
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[[
            InlineKeyboardButton(text=button_text, callback_data=callback_data)
        ]]
    )
    
    try:
        # Send message with or without image
        if image_path and os.path.exists(image_path):
            await bot.send_photo(
                chat_id=user_id,
                photo=FSInputFile(image_path),
                caption=text,
                parse_mode=parse_mode,
                reply_markup=keyboard
            )
        else:
            await bot.send_message(
                chat_id=user_id,
                text=text,
                parse_mode=parse_mode,
                reply_markup=keyboard
            )

        # Wait for user confirmation
        future = asyncio.get_event_loop().create_future()
        waiting_users[user_id] = future
        await future
        
    except asyncio.CancelledError:
        print(f"Instruction step cancelled for user {user_id}")
    finally:
        waiting_users.pop(user_id, None)

@dp.callback_query(lambda c: c.data and c.data.startswith("next_"))
async def handle_next_button(callback_query: CallbackQuery):
    """Handle 'Next' button press in instruction flow."""
    user_id = callback_query.from_user.id
    await callback_query.answer()
    await callback_query.message.edit_reply_markup(reply_markup=None)
    
    future = waiting_users.get(user_id)
    if future and not future.done():
        future.set_result(True)

@dp.callback_query(lambda c: c.data == "instruction")
async def handle_instruction(callback_query: CallbackQuery):
    """Handle instruction button press."""
    await callback_query.answer()
    await bot.delete_message(
        chat_id=callback_query.message.chat.id, 
        message_id=callback_query.message.message_id
    )
    await show_instruction(callback_query.message, user_id=callback_query.from_user.id)

@dp.message(F.text.startswith('/instruction'))
async def show_instruction(message: Message, user_id=None):
    """Show complete instruction sequence to user."""
    user_id = user_id if user_id else message.from_user.id
    active_users = get_active_users()
    
    active_users.add(user_id)
    try:
        # Show downloading instructions
        await show_downloading_instructions(message, user_id)
        
        # Show usage instructions with images
        await show_usage_instructions(message, user_id)
        
        # Show completion message
        await message.answer("Инструкция завершена! Теперь вы можете пользоваться ботом")
        
    finally:
        active_users.discard(user_id)
    
    # Show main menu after completion
    await show_menu_to_user(message, user_id)

async def show_downloading_instructions(message: Message, user_id: int):
    """Show downloading instructions sequence."""
    # Initial instructions without button
    await message.answer(DOWNLOADING_INSTRUCTIONS[0], parse_mode="Markdown")
    await asyncio.sleep(4.5)
    await message.answer(DOWNLOADING_INSTRUCTIONS[1], parse_mode="Markdown")
    await asyncio.sleep(0.3)
    await message.answer(DOWNLOADING_INSTRUCTIONS[2], parse_mode="Markdown")

    # Final downloading instruction with confirmation
    await show_instruction_step(
        user_id=user_id,
        text=DOWNLOADING_INSTRUCTIONS[3],
        parse_mode="Markdown",
        button_text="Готово"
    )

async def show_usage_instructions(message: Message, user_id: int):
    """Show usage instructions sequence with images."""
    # First usage instruction without image
    await show_instruction_step(
        user_id=user_id,
        text=USAGE_INSTRUCTIONS[0],
        parse_mode="Markdown"
    )

    # Instructions with images
    for i, instruction in enumerate(USAGE_INSTRUCTIONS[1:-1]):
        image_path = os.path.join(INSTRUCTION_IMAGES_PATH, f"{i}.jpg")
        await show_instruction_step(
            user_id=user_id,
            text=instruction,
            image_path=image_path,
            parse_mode="Markdown"
        )

    # Final instruction
    await show_instruction_step(
        user_id=user_id,
        text=USAGE_INSTRUCTIONS[-1],
        parse_mode="Markdown",
        button_text="Все понятно"
    )
