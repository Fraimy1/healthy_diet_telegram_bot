from aiogram import F
from aiogram.types import Message, InlineKeyboardButton, InlineKeyboardMarkup
import asyncio

from bot.bot_init import bot, dp
from utils.utils import create_user_profile
from config.config import WELCOME_MESSAGES

# Track users in instruction flow
active_users_instruction = set()

@dp.message(F.text == "/start")
async def cmd_start(message: Message, user_id=None):
    """
    Handle the /start command.
    Creates user profile if needed and shows welcome message with instruction options.
    """
    user_id = user_id if user_id else message.from_user.id 

    # Create user profile in database
    create_user_profile(user_id, message.from_user.username)
    
    # Prevent multiple instruction flows
    if user_id in active_users_instruction:
        await message.answer("Инструкция уже выведена на экран. Пожалуйста, завершите её перед началом новой.")
        return

    # Send welcome messages sequence
    await send_welcome_sequence(message)

async def send_welcome_sequence(message: Message):
    """Helper function to send the welcome message sequence with delays"""
    await message.answer(WELCOME_MESSAGES[0], parse_mode="Markdown")
    await asyncio.sleep(0.5)
    await message.answer(WELCOME_MESSAGES[1], parse_mode="Markdown")
    await asyncio.sleep(0.7)
    await message.answer(WELCOME_MESSAGES[2], parse_mode="Markdown")
    await asyncio.sleep(2.5)

    # Create instruction choice keyboard
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Пройти инструкцию",
                    callback_data="instruction"
                )
            ],
            [
                InlineKeyboardButton(
                    text="Пропустить инструкцию",
                    callback_data="main_menu"
                )
            ]
        ]
    )
    
    await message.answer(WELCOME_MESSAGES[3], parse_mode="Markdown", reply_markup=keyboard)

# Export active_users_instruction set for use in other modules
def get_active_users():
    """Get the set of users currently in instruction flow"""
    return active_users_instruction
