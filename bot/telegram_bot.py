"""
This Python script defines the functionality of a Telegram bot that helps users analyze receipts.

- Model and Database Initialization:
    - Loads the pre-trained model for text classification (bert_2level_model) and label encoder (le).
    - Initializes the database connection and retrieves essential information like database path and unique receipt IDs.
    - Restructures the microelements data for efficient processing.

- Bot Configuration:
    - Defines Telegram bot token (TELEGRAM_TOKEN) for authentication.
    - Sets up the bot instance (bot) and dispatcher (dp) for handling user interactions.

- Data Paths and Resources:
    - Specifies paths for instruction images (INSTRUCTION_IMAGES_PATH).
    - Defines text messages for downloading and usage instructions (DOWNLOADING_INSTRUCTIONS, USAGE_INSTRUCTIONS).
    - Sets paths for configuration files (DATABASE_PATH, config.py).
    - Imports helper functions from other modules (data_processor, utils).

- Dynamic Data Structures:
    * Uses dictionaries to store dynamic information:
        - waiting_users: Tracks users currently waiting for button presses in the instruction.
        - active_users_instruction: Keeps track of users currently going through the instruction.
        - set_confidence_markup_queue: Stores message IDs for later deletion during confidence threshold change.
        - pending_receipts: (In-memory storage) Stores receipt IDs awaiting processing.

- User Interface and Interaction:

    - /start command:
        - Creates a user profile if it doesn't exist.
        - Checks if the user is already in instruction or help flow.
        - Welcomes the user and presents a button to start the instruction.

    - /instruction command or "   " button:
        - Guides the user through the instruction step-by-step using the show_instruction_step function.
        - Presents downloading instructions first, followed by detailed usage instructions with images.
        - Provides a completion message and shows the main menu after finishing.

    - Confidence Threshold Change:
        - Allows users to adjust the minimum confidence threshold for predictions using the /confidence command.
        - Presents options to enter a new value or skip the change.
        - Saves the updated confidence threshold in the user profile.

    - Main Menu Options:
        - Users can access various functionalities through the main menu (implicitly presented after interactions).
        - These functionalities might include adding receipts to history, receiving Excel reports, etc. (not implemented in this snippet).

- Helper Functions:

    - show_instruction_step: Sends a message with optional image and a button, waiting for user interaction.
    - handle_next_button: Acknowledges button press, removes keyboard, and resumes the waiting coroutine in instruction flow.
    - cmd_start: Handles the /start command, creates user profile, checks user flow, and presents welcome messages.
    - handle_instruction: Handles the "   " button, acknowledges the press, removes the keyboard, and starts the instruction flow.
    - set_confidence: Presents messages for users to input a new confidence threshold or skip the change.
    - handle_skip_confidence_change: Acknowledges skipping confidence change, removes previous messages, and informs the user.
    - handle_set_confidence: Processes user input for confidence value, updates the user profile, and sends a confirmation message.
    - set_menu_button: Defines available commands for the bot upon startup.
    - add_to_history: Handles the /add_to_history command, toggles the "add to history" option in the user profile, and sends a confirmation message.
    - set_return_excel_document: Handles the /return_excel_document command, toggles the "return Excel document" option in the user profile, and sends a confirmation message.
    - send_excel_document (not implemented): Formats data, creates an Excel file with predictions for a receipt, and sends it to the user.

Note: This script snippet focuses on core functionalities. Additional functionalities like receipt processing or history management might be implemented in separate modules.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
import uuid
from collections import defaultdict
import tempfile
import uuid

from aiogram import Bot, Dispatcher, F
from aiogram.types import (
    BotCommand,
    CallbackQuery,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from model.model_init import initialize_model
from utils.data_processor import predict_product_categories, save_data_for_database, get_sorted_user_receipts, count_product_amounts, restructure_microelements, get_microelements_data, parse_json
from utils.utils import create_user_profile, create_back_button, update_user_contributions
from utils.db_utils import db_connection, make_backup
from config.config import (
    DATABASE_PATH, 
    INSTRUCTION_IMAGES_PATH, 
    DOWNLOADING_INSTRUCTIONS, 
    WELCOME_MESSAGES, 
    USAGE_INSTRUCTIONS,
    TELEGRAM_TOKEN,
    HELP_TEXT,
    ABBREVIATION_MICROELEMENTS_DICT
)

# ======= Bot initialization =======
bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

# Dynamic structures
waiting_users = {}
active_users_instruction = set()
set_confidence_markup_queue = {}
# ======= Model initialization =======
bert_2level_model, le = initialize_model()

# ======= Database initialization =======
MICROELEMENTS_TABLE = restructure_microelements()

# ======= Bot main functionality =======

async def show_instruction_step(user_id, text, image_path=None, parse_mode='Markdown', button_text='Далее'):
    """
    Send a step of instruction to a user.

    This function sends a message to a user with a "Next" button.
    If an image path is given, it is sent as a photo.
    The function then waits for the user to press the "Next" button.
    When the user presses the button, the function returns.
    If the function is cancelled before the user presses the button, it will
    gracefully handle the cancellation and clean up after itself.

    :param user_id: The Telegram user ID to send the instruction to.
    :param text: The text of the instruction to send.
    :param image_path: The path to an image to send as a photo. If None, a text message will be sent instead.
    :param parse_mode: The parse mode to use for the message. Default is 'Markdown'.
    :param button_text: The text on the "Next" button. Default is 'Далее'.
    """
    callback_data = f"next_{user_id}"
    # Create the "Next" button
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=button_text, callback_data=callback_data)]
        ]
    )
    # Send the instruction with or without image
    try:
        if image_path and os.path.exists(image_path):
            photo_file = FSInputFile(image_path)
            await bot.send_photo(
                chat_id=user_id,
                photo=photo_file,
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

        # Wait for the user to press "Next"
        future = asyncio.get_event_loop().create_future()
        waiting_users[user_id] = future
        await future
    except asyncio.CancelledError:
        # Gracefully handle cancellation
        print(f"Task for user {user_id} was cancelled.")
    finally:
        # Ensure cleanup
        waiting_users.pop(user_id, None)

@dp.callback_query(lambda c: c.data and c.data.startswith("next_"))
async def handle_next_button(callback_query: CallbackQuery):
    """
    Handle the "Next" button press in the instruction.

    This function is called when the user presses the "Next" button in the instruction.
    It acknowledges the button press, removes the keyboard and resumes the coroutine that
    was waiting for the button press.
    """
    user_id = callback_query.from_user.id

    # Acknowledge the button press
    await callback_query.answer()

    # Remove the keyboard
    await callback_query.message.edit_reply_markup(reply_markup=None)

    # Resume the coroutine
    future = waiting_users.get(user_id)
    if future and not future.done():
        future.set_result(True)

@dp.message(F.text == "/start")
async def cmd_start(message: Message, user_id=None):
    """
    Handle the /start command.

    This function is called when the user sends the /start command.
    It creates a user profile if it doesn't exist already, checks if the user is already running /instruction or /help,
    and then prints a welcome message with a button to start the instruction.
    """
    user_id = user_id if user_id else message.from_user.id 

    create_user_profile(user_id, message.from_user.username)
    
    # Check if the user is already running /instruction or /help
    if user_id in active_users_instruction:
        await message.answer("Инструкция уже выведена на экран. Пожалуйста, завершите её перед началом новой.")
        return

    # Keep this section as it is
    await message.answer(WELCOME_MESSAGES[0], parse_mode="Markdown")
    await asyncio.sleep(0.5)
    await message.answer(WELCOME_MESSAGES[1], parse_mode="Markdown")
    await asyncio.sleep(0.7)
    await message.answer(WELCOME_MESSAGES[2], parse_mode="Markdown")
    await asyncio.sleep(2.5)
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="Пройти инструкцию",callback_data="instruction")
            ],
            [
                InlineKeyboardButton(text="Пропустить инструкцию",callback_data="main_menu")
            ]
        ]
    )
    await message.answer(WELCOME_MESSAGES[3], parse_mode="Markdown", reply_markup=keyboard)

@dp.callback_query(lambda c: c.data and c.data.startswith("instruction"))
async def handle_instruction(callback_query: CallbackQuery):
    """
    Handle a user clicking on the 'instruction' button.

    This function is called when the user presses the "instruction" button.
    It acknowledges the button press, removes the keyboard and shows the instruction.
    """
    await callback_query.answer()
    await bot.delete_message(chat_id=callback_query.message.chat.id, message_id=callback_query.message.message_id)
    await show_instruction(callback_query.message, user_id = callback_query.from_user.id)

@dp.message(F.text.startswith('/instruction'))
async def show_instruction(message, user_id=None):
    """
    Shows the instruction to a user.

    This function is called when the user types /instruction.
    It shows the instruction to the user, step by step, using the show_instruction_step helper function.
    The instruction is divided into two parts: downloading instructions and usage instructions.
    The downloading instructions are shown first, and then the usage instructions are shown.
    The usage instructions are shown with images, and the user is asked to press the "Next" button to proceed.
    When the instruction is complete, a completion message is sent and the main menu is shown.
    """
    user_id = user_id if user_id else message.from_user.id 
    active_users_instruction.add(user_id)
    try:
        await message.answer(DOWNLOADING_INSTRUCTIONS[0], parse_mode="Markdown")
        await asyncio.sleep(4.5)
        await message.answer(DOWNLOADING_INSTRUCTIONS[1], parse_mode="Markdown")
        await asyncio.sleep(0.3)
        await message.answer(DOWNLOADING_INSTRUCTIONS[2], parse_mode="Markdown")

        # Final downloading instruction with "Готово" button
        await show_instruction_step(
            user_id=user_id,
            text=DOWNLOADING_INSTRUCTIONS[3],
            parse_mode="Markdown",
            button_text="Готово"
        )

        # Iterate through usage instructions with images
        await show_instruction_step(
            user_id=user_id,
            text=USAGE_INSTRUCTIONS[0],
            parse_mode="Markdown"
        )

        for i, instruction in enumerate(USAGE_INSTRUCTIONS[1:-1]):
            image_path = os.path.join(INSTRUCTION_IMAGES_PATH, f"{i}.jpg")
            await show_instruction_step(
                user_id=user_id,
                text=instruction,
                image_path=image_path,
                parse_mode="Markdown"
            )

        # Show final usage instruction
        await show_instruction_step(
            user_id=user_id,
            text=USAGE_INSTRUCTIONS[-1],
            parse_mode="Markdown",
            button_text="Все понятно"
        )

        # Send a completion message
        await message.answer("Инструкция завершена! Теперь вы можете пользоваться ботом")

    finally:
        # Remove the user from the active_users_instruction set when the flow ends
        active_users_instruction.discard(user_id)

    # Show the main menu after the instruction is complete
    await show_main_menu(message, user_id)

async def set_confidence(message: Message, user_id):
    """
    Ask the user to input a new confidence threshold.

    This function sends two messages, asking the user to input a number between 1 and 99.
    The first message explains the purpose of the confidence threshold, and the second message asks the user to input a number.
    The user can also choose to skip changing the confidence threshold by pressing a button.
    The messages and the button are stored in `set_confidence_markup_queue` for later deletion.
    """
    user_id = user_id if user_id else message.from_user.id 
    first_message = await message.answer(
        'Вы можете изменить минимальную уверенность модели, отправив число от 1 до 99 (в процентах).\n\n'
        'Например, "35" установит минимальную уверенность на 35%. Предсказания с меньшей уверенностью будут считаться нераспознанными.\n'
        'Рекомендуем выбирать значение от 20% до 85%.\n'
        'Подробнее читайте в /help.',
        parse_mode='Markdown'
    )

    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Пропустить изменение уверенности модели",
                    callback_data=f"skip_confidence_change_{user_id}"
                )
            ]
        ]
    )
    second_message = await message.answer('Отправьте целое число от 1 до 100 своим следующим сообщением:', reply_markup=keyboard)
    
    # Store message IDs for later deletion
    set_confidence_markup_queue[user_id] = {
        'first_message': first_message.message_id,
        'second_message': second_message.message_id
    }

@dp.callback_query(lambda c: c.data and c.data.startswith("skip_confidence_change_"))
async def handle_skip_confidence_change(callback_query: CallbackQuery):
    """
    Handle the "Skip changing the confidence threshold" button press in the instruction.

    This function is called when the user presses the "Skip changing the confidence threshold" button in the instruction.
    It acknowledges the button press, removes the previous messages, removes the stored message IDs, and informs the user that the confidence change was skipped.
    """
    user_id = callback_query.from_user.id

    # Acknowledge the callback
    await callback_query.answer()

    # Delete the previous messages
    message_ids = set_confidence_markup_queue.get(user_id, {})
    for msg_id in message_ids.values():
        await bot.delete_message(chat_id=user_id, message_id=msg_id)

    # Remove the stored message IDs
    set_confidence_markup_queue.pop(user_id, None)

    # Inform the user that the confidence change was skipped
    await callback_query.message.answer("Изменение минимальной уверенности было пропущено. Вы можете изменить её позже через команду /confidence.")

    await show_main_menu(callback_query.message, user_id)

@dp.message(lambda message: message.text and message.text.isdigit() and 1 <= int(message.text) <= 99)
async def handle_set_confidence(message: Message, user_id=None):
    """
    Handle a message with a confidence value sent by the user.

    This function is called when the user sends a message with a whole number between 1 and 99 (inclusive) as text.
    It deletes the previous messages, removes the stored message IDs, loads the user profile, updates the minimal confidence,
    saves the user profile, and sends a message back to the user with a success message and a back button.

    Args:
        message (Message): The message sent by the user.
        user_id (int, optional): The user ID of the user that sent the message. Defaults to None.
    """
    user_id = user_id if user_id else message.from_user.id 
    display_confidence_value = float(message.text)
    confidence_value = float(message.text) / 100

    # Delete the previous messages
    message_ids = set_confidence_markup_queue.get(user_id, {})
    for msg_id in message_ids.values():
        await bot.delete_message(chat_id=user_id, message_id=msg_id)

    # Remove the stored message IDs
    set_confidence_markup_queue.pop(user_id, None)

    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE main_database SET minimal_confidence_for_prediction = ? WHERE user_id = ?", (confidence_value, user_id))
        conn.commit()

    back_button = create_back_button(text="Назад", callback_data="main_menu")
    await message.answer(f"Минимальная уверенность модели установлена на {display_confidence_value}%.", reply_markup=back_button)

@dp.startup()
async def set_menu_button(bot: Bot):
    """
    Set the menu commands for the bot.

    This function is called when the bot is started, and sets the menu commands that users can use to interact with the bot.
    The commands are as follows:
    - /start: Start the bot / Hello message
    - /show_menu: Show the main menu
    - /instruction: Instruction on how to use the bot
    - /help: All information about the bot
    - /confidence: Change the minimal confidence for prediction
    - /add_to_history: Add receipts to your history?
    - /return_excel_document: Return Excel documents with predictions on each receipt?

    Args:
        bot (Bot): The bot instance

    Returns:
        None
    """
    main_menu_commands = [
        BotCommand(command='/start', description='Запуск бота / Приветствие'),
        BotCommand(command='/show_menu', description='Главное меню'),
        BotCommand(command='/instruction', description='Инструкция по использованию'),
        BotCommand(command='/help', description='Вся информация о боте'),
        BotCommand(command='/confidence', description='Поменять уверенность предсказания модели'),
        BotCommand(command='/add_to_history', description='Добавлять ли чеки в вашу историю?'),
        BotCommand(command='/return_excel_document', description='Отправлять ли Excel документы с предсказаниями на каждый чек?'),
    ]
    await bot.set_my_commands(main_menu_commands)

@dp.message(F.text.startswith('/add_to_history'))    
async def add_to_history(message:Message, user_id):
    """
    Handle the /add_to_history command.

    This function is called when the user sends the /add_to_history command.
    It toggles the 'add_to_history' option in the user's profile and sends a message back to the user with a success message and a back button.

    Args:
        message (Message): The message sent by the user.
        user_id (int, optional): The user ID of the user that sent the message. Defaults to None.
    """
    user_id = user_id if user_id else message.from_user.id 

    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT add_to_history FROM main_database WHERE user_id = ?", (user_id,))
        add_to_history_value = not cursor.fetchone()[0]
        cursor.execute("UPDATE main_database SET add_to_history = ? WHERE user_id = ?", (add_to_history_value, user_id))
        conn.commit()

    back_button = create_back_button(text="Главное меню", callback_data="main_menu")
    
    if add_to_history_value:
        await message.answer("✅ Добавление чеков в вашу историю включено.", reply_markup=back_button)
    else:
        await message.answer("❌ Добавление чеков в вашу историю отключено.", reply_markup=back_button)

@dp.message(F.text.startswith('/return_excel_document'))    
async def set_return_excel_document(message:Message, user_id=None):
    """
    Handle the /return_excel_document command.

    This function is called when the user sends the /return_excel_document command.
    It toggles the 'return_excel_document' option in the user's profile and sends a message back to the user with a success message and a back button.

    Args:
        message (Message): The message sent by the user.
        user_id (int, optional): The user ID of the user that sent the message. Defaults to None.
    """
    user_id = user_id if user_id else message.from_user.id 

    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT return_excel_document FROM main_database WHERE user_id = ?", (user_id,))
        return_excel_document_value = not cursor.fetchone()[0]
        cursor.execute("UPDATE main_database SET return_excel_document = ? WHERE user_id = ?", (return_excel_document_value, user_id))
        conn.commit()

    back_button = create_back_button(text="Главное меню", callback_data="main_menu")
    
    if return_excel_document_value:
        await message.answer("✅ Отправление Excel документов с предсказаниями на каждый чек включено.", reply_markup=back_button)
    else:
        await message.answer("❌ Отправление Excel документов с предсказаниями на каждый чек отключено.", reply_markup=back_button)

# In-Memory Storage for pending receipt IDs
pending_receipts = {}

async def send_excel_document(data_recieved, receipt_info, user_id):
    """
    Send an Excel document to the user.

    This function is called when the user has requested an Excel document with predictions on a receipt.
    It formats the data, adds new columns, splits the amount_unit column into amount and unit,
    sets the percentage column, removes unnecessary columns, reorders the columns,
    creates a temporary file, saves the DataFrame to the temporary Excel file,
    creates a filename for the Excel file, sends the document to the user,
    and removes the temporary file.

    Args:
        data_recieved (DataFrame): The DataFrame to be formatted and sent to the user.
        receipt_info (dict): A dictionary containing information about the receipt.
        user_id (int): The user ID of the user to send the document to.

    Returns:
        None
    """
    try:
        # Format the data
        data = data_recieved.copy()
        data.fillna('n/a', inplace=True)

        # Add new columns
        data['amount_unit'] = data.get('amount', ['n/a'] * len(data))
        data['receipt_id'] = receipt_info.get('receipt_id', uuid.uuid4().hex())
        data['unique_id'] = [str(uuid.uuid4()) for _ in range(len(data))]

        # Split amount_unit into amount and unit
        data['amount'] = data['amount_unit'].apply(lambda x: x.split()[0] if x != 'n/a' else 'n/a')
        data['unit'] = data['amount_unit'].apply(lambda x: x.split()[1] if x != 'n/a' else 'n/a')

        # Set percentage column
        data['percentage'] = data.get('percentage', ['n/a'] * len(data))

        # Remove unnecessary columns
        data.drop(columns=['sum_rub', 'original_entry', 'prediction'], inplace=True, errors='ignore')

        # Reorder columns
        ordered_columns = ['receipt_id', 'unique_id', 'name', 'product_name', 'user_prediction', 
                           'amount_unit', 'amount', 'unit', 'percentage', 'confidence']
        data = data[ordered_columns]

        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
            temp_path = tmp.name
            # Save the DataFrame to the temporary Excel file
            data.to_excel(temp_path, sheet_name='Receipt', index=False)

        # Create a filename for the Excel file
        filename = f"receipt_{receipt_info['receipt_id']}.xlsx"

        # Send the document
        await bot.send_document(
            chat_id=user_id,
            document=FSInputFile(temp_path, filename=filename),
            caption=f'Чек от {receipt_info["purchase_datetime"]}'
        )

        # Remove the temporary file
        os.unlink(temp_path)

    except Exception as e:
        logging.error(f"Error sending Excel document: {str(e)}")
        await bot.send_message(
            chat_id=user_id,
            text=f"Извините, произошла ошибка при создании Excel-файла: {str(e)}"
        )

async def process_receipt(receipt_data, user_id, user_folder, idx_info=""):
    """
    Processes a receipt and updates the database accordingly.

    The function first tries to extract relevant information from the receipt data using the `predict_product_categories` function. 
    It then saves the receipt data to a file and checks if the receipt is unique. 
    If the receipt is unique, it adds the receipt data to the user's history and the dataset. 
    If the receipt is not unique, it asks the user whether they want to add the receipt to their history or not.

    Args:
        receipt_data (dict): A dictionary containing the receipt data.
        user_id (int): The user ID of the user who uploaded the receipt.
        user_folder (str): The path to the user's folder in the database.

    Returns:
        None
    """
    receipt_id = None  # Initialize receipt_id outside the try block
    
    try:
        with db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT minimal_confidence_for_prediction, add_to_history, return_excel_document FROM main_database WHERE user_id = ?", (user_id,))
            user_data = cursor.fetchone()
            user_data = dict(user_data) if user_data else {} 
            unique_receipt_ids = cursor.execute("SELECT DISTINCT receipt_id FROM user_purchases WHERE user_id = ?", (user_id,)).fetchall()
        
        unique_receipt_ids = {row[0] for row in unique_receipt_ids}
        min_confidence = user_data.get('minimal_confidence_for_prediction', 0.5)
        add_to_history_bool = user_data.get('add_to_history', False)        
        return_excel_document = user_data.get('return_excel_document', False)

        items_data, receipt_info = parse_json(receipt_data)
        receipt_id = receipt_info.get('receipt_id')  # Now receipt_id is defined
        display_date = datetime.strptime(receipt_info.get('purchase_datetime'), '%Y-%m-%d_%H:%M:%S').strftime('%d.%m.%Y %H:%M')
        data = predict_product_categories(items_data, bert_2level_model, le, min_confidence)
        
        # Deprecated
        receipt_file_path = os.path.join(user_folder, f'file_{receipt_id}.json')
        with open(receipt_file_path, 'w') as f:
            json.dump(receipt_data, f, ensure_ascii=False, indent=4)
        # ===

        if return_excel_document:
                await send_excel_document(data, receipt_info, user_id)
        
        if receipt_id not in unique_receipt_ids:
            save_data_for_database(data, receipt_info, user_id)
            
            purchase_date = datetime.strptime(receipt_info['purchase_datetime'], '%Y-%m-%d_%H:%M:%S').strftime('%d.%m.%Y %H:%M')
            if add_to_history_bool:
                await bot.send_message(user_id, f"✅ {idx_info} Чек от {purchase_date} добавлен в датасет и вашу историю")
            else:
                await bot.send_message(user_id, f"➡️ {idx_info} Чек от {purchase_date} был добавлен в общий датасет, но не был добавлен в вашу историю исходя из ваших настроек")  
        else:
            await bot.send_message(
                user_id,
                text = (f"⚠️ {idx_info} Вы уже добавляли этот чек.\n"
                        f'Дата покупки: {display_date}\n'
                        f'ID чека: {receipt_id}\n'
                        "Он будет пропущен")
                )

    except Exception as e:
        error_msg = f"Error processing receipt {receipt_id if receipt_id else 'unknown'} for user {user_id}: {e}"
        print(error_msg)
        await bot.send_message(user_id, text = f"Произошла ошибка при обработке чека{' с ID ' + receipt_id if receipt_id else ''}")


@dp.message(lambda message: message.document and message.document.file_name.endswith('.json'))
async def handle_json_upload(message: Message):
    """
    Handle a JSON file upload from a user.

    The function first saves the file to the user's folder in the database,
    then processes the JSON data in the file, calling the process_receipt
    function for each receipt in the file. Then it removes the file and shows
    the main menu.

    :param message: The message containing the file
    """
    file_name = f'file_{datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}.json'
    user_id = message.from_user.id
    
    user_folder = os.path.join(DATABASE_PATH, f'user_{user_id}')
    os.makedirs(user_folder, exist_ok=True)
    file_path = os.path.join(user_folder, file_name)
    
    try:
        # Download and save the file
        await bot.download(message.document, destination=file_path)
        with open(file_path, 'r') as f:
            data_received = json.load(f)
        
        idx_info = ''
        if isinstance(data_received, list):
            for i, receipt_data in enumerate(data_received, start=1):
                idx_info = f'{i}/{len(data_received)}'
                await process_receipt(receipt_data, user_id, user_folder, idx_info)
        else:
            await process_receipt(data_received, user_id, user_folder, idx_info)
        
    except Exception as e:
        await message.answer("Произошла ошибка при обработке файла. Попробуйте еще раз")
        print(f"Error processing file for user {user_id}: {e}")

    finally:
        os.remove(file_path)
        # Show the main menu after processing the upload
        back_button = create_back_button(text="Главное меню", callback_data="main_menu")
        await message.answer("Все файлы были обработаны", reply_markup=back_button)
        # Updating user's contribution to dataset
        with db_connection() as conn:
            update_user_contributions(user_id, conn)

@dp.message(F.text.startswith('/help'))
async def help_command(message: Message):
    """
    Handle the /help command.

    This function is called when a user sends a message that starts with /help.
    It sends a message with the help text and a "Back to main menu" button.

    :param message: The message containing the /help command
    """
    back_button = create_back_button(text="Главное меню", callback_data="main_menu")
    await message.answer(HELP_TEXT, parse_mode='Markdown', reply_markup=back_button)

# ======= Main menu =======

@dp.message(F.text.startswith('/show_menu'))
async def show_main_menu(message: Message, user_id=None):
    """
    Handles the /show_menu command.

    This function is called when a user sends a message that starts with /show_menu.
    It deletes the previous message, loads the user profile, and sends a message with the main menu
    containing buttons with the current values of the minimal confidence, add_to_history, and return_excel_document options.

    :param message: The message containing the /show_menu command
    :param user_id: The user ID of the user that sent the message (optional)
    """
    user_id = user_id if user_id else message.from_user.id
    try:
        await bot.delete_message(chat_id=message.chat.id, message_id=message.message_id)
    except:
        pass
    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT minimal_confidence_for_prediction, add_to_history, return_excel_document FROM main_database WHERE user_id = ?", (user_id,))
        user_data = cursor.fetchone()
        user_data = dict(user_data) if user_data else {}
    
    min_confidence = int(user_data.get('minimal_confidence_for_prediction', 0.5) * 100)
    add_to_history_emoji = '✅' if user_data.get('add_to_history', False) else '❌'
    return_excel_document = '✅' if user_data.get('return_excel_document', False) else '❌'
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="📖 Инструкция", callback_data="menu_instruction"),
                InlineKeyboardButton(text="❓ Помощь", callback_data="menu_help")
            ],
            [
                InlineKeyboardButton(text=f"⚙️ Изменить уверенность - {min_confidence}%", callback_data="menu_confidence"),
            ],
            [
                InlineKeyboardButton(text=f"📝 Добавлять в историю {add_to_history_emoji}", callback_data="menu_add_to_history"),
            ],
            [
                InlineKeyboardButton(text=f" 📊 Получать Excel документ {return_excel_document}", callback_data="menu_return_excel_document"),
            ],
            [            
                InlineKeyboardButton(text="📜 История покупок", callback_data="menu_history"),
            ]
        ]
    )  
    await message.answer("Меню", reply_markup=keyboard)


@dp.callback_query(lambda c: c.data and c.data.startswith("menu_"))
async def handle_menu_callbacks(callback_query: CallbackQuery):
    """
    Handles callback queries with the "menu_" prefix.

    It acknowledges the callback to prevent loading icons,
    removes the menu message and options,
    and then handles the menu options.

    :param callback_query: The callback query containing the user ID and the menu option
    """
    data = callback_query.data
    # Acknowledge the callback to prevent loading icons
    await callback_query.answer()

    # Remove the menu message and options
    try: 
        await bot.delete_message(chat_id=callback_query.message.chat.id, message_id=callback_query.message.message_id)
    except:
        pass
    
    user_id = callback_query.from_user.id
    # Handle menu options
    if data == "menu_start":
        await cmd_start(callback_query.message, user_id)
    elif data == "menu_instruction":
        await show_instruction(callback_query.message, user_id)
    elif data == "menu_help":
        await help_command(callback_query.message)
    elif data == "menu_confidence":
        await set_confidence(callback_query.message, user_id)
    elif data == "menu_history":
        await show_history_options(callback_query.message)
    elif data == "menu_add_to_history":
        await add_to_history(callback_query.message, user_id)
    elif data == "menu_return_excel_document":
        await set_return_excel_document(callback_query.message, user_id)

# ======= History view =======

@dp.message(F.text.startswith('/show_history'))
async def show_history_options(message: Message):
    """
    Handles the /show_history command.

    This function is called when a user sends a message that starts with /show_history.
    It deletes the previous message, sends a message with the history options containing buttons
    with the text "Чеки", "Профиль", "Категории продуктов", "Микроэлементы", and "Выйти".
    """
    
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="🧾 Чеки", callback_data="history_receipts"),
                InlineKeyboardButton(text="📊 Профиль", callback_data="history_user_profile"),
            ],
            [
                InlineKeyboardButton(text="🍎 Категории продуктов", callback_data="history_categories"),
            ],
            [
                InlineKeyboardButton(text="💊 Микроэлементы", callback_data="history_microelements"),
            ],
            [
                InlineKeyboardButton(text="🚪 Выйти", callback_data="history_main_menu"),
            ]
        ]
    )  
    await message.answer('📜 История покупок', reply_markup=keyboard)

@dp.callback_query(lambda c: c.data and c.data.endswith("main_menu"))
async def handle_history_quit(callback_query: CallbackQuery):
    """
    Handles the callback query with the "main_menu" suffix.

    Deletes the menu message and sends the main menu to the user.

    :param callback_query: The callback query containing the user ID
    """
    user_id = callback_query.from_user.id
    await callback_query.answer()
    await bot.delete_message(chat_id=callback_query.message.chat.id, message_id=callback_query.message.message_id)
    await show_main_menu(callback_query.message, user_id)

@dp.callback_query(lambda c: c.data and c.data.startswith("back_to_history"))
async def handle_back_to_history(callback_query: CallbackQuery):
    """
    Handles the callback query with the "back_to_history" prefix.

    Acknowledges the callback to prevent loading icons,
    removes the menu message and options,
    and then shows the history options again.
    If the callback query data ends with "delete", it also deletes the menu message.

    :param callback_query: The callback query containing the user ID
    """
    await callback_query.answer()
    await callback_query.message.edit_reply_markup(reply_markup=None)
    if callback_query.data.endswith("delete"):
        await bot.delete_message(chat_id=callback_query.message.chat.id, message_id=callback_query.message.message_id)

    await show_history_options(callback_query.message)


# ======= User profile =======

@dp.callback_query(lambda c: c.data and c.data == "history_user_profile")
async def handle_user_profile(callback_query: CallbackQuery):
    await callback_query.answer()  
    await bot.delete_message(chat_id=callback_query.message.chat.id, message_id=callback_query.message.message_id)
    user_id = callback_query.from_user.id

    with db_connection() as conn:        
        cursor = conn.cursor()
        cursor.execute("""
            SELECT user_id, user_name, registration_date, add_to_history, original_receipts_added, products_added, minimal_confidence_for_prediction
            FROM main_database
            WHERE user_id = ?
        """, (user_id,))
        user_data = cursor.fetchone()
        user_data = dict(user_data) if user_data else {}

        cursor.execute("""
            SELECT COUNT(DISTINCT receipt_id) 
            FROM user_purchases
            WHERE user_id = ?
        """, (user_id,))
        receipts_count = cursor.fetchone()['COUNT(DISTINCT receipt_id)']

        cursor.execute("""
            SELECT COUNT(DISTINCT receipt_id) 
            FROM user_purchases
            WHERE user_id = ? AND in_history = 1
        """, (user_id,))
        receipts_count_in_history = cursor.fetchone()['COUNT(DISTINCT receipt_id)']

        cursor.execute("""
            SELECT SUM(total_sum) 
            FROM user_purchases
            WHERE user_id = ? AND in_history = 1
        """, (user_id,))
        total_sum = cursor.fetchone()['SUM(total_sum)'] or 0

        # products_added is now properly stored in the main_database after the update
        # No need to recalculate it again, just use user_data['products_added']

    registration_date = user_data['registration_date']

    statistics = (
        f"📊 Общая статистика:\n\n"
        
        f"📝 Всего чеков добавлено: {receipts_count}\n"
        f"📈 Уникальных чеков добавлено: {user_data.get('original_receipts_added', 0)}\n"
        f"📈 Чеков в истории: {receipts_count_in_history}\n"
        f"🍔 Продуктов добавлено: {user_data.get('products_added', 0)}\n"
        f"💸 Всего денег потрачено: {int(total_sum)}₽\n\n"

        f"🕰️ Дата регистрации: {registration_date}\n"
        f"💯 Минимальная уверенность модели: {int(user_data.get('minimal_confidence_for_prediction', 0.5)*100)}%"
    )
    
    back_button = create_back_button(text="К истории покупок", callback_data="back_to_history_delete")

    await bot.send_message(user_id, statistics, reply_markup=back_button)


# ======= Receipts Interface =======

async def display_receipts(user_id, message, receipts, page=0):
    """
    Displays a list of receipts to the user with pagination.

    Args:
    user_id (int): The user ID to send the message to.
    message (aiogram.types.Message): The original message to edit.
    receipts (list): A list of receipts to display.
    page (int, optional): The page number to display. Defaults to 0.

    Returns:
    InlineKeyboardMarkup: The markup that was used to send the message.
    """
    keyboard = []
    start_index = page * 7
    end_index = min(start_index + 7, len(receipts))

    for i in range(start_index, end_index):
        receipt = receipts[i]
        date = datetime.strptime(receipt['purchase_datetime'], '%Y-%m-%d %H:%M:%S')
        display_date = date.strftime('%d.%m.%Y %H:%M')
        keyboard.append([InlineKeyboardButton(text=f"Чек от {display_date}", callback_data=f"display_receipts_receipt_{receipt['receipt_id']}")])

    nav_buttons = []
    if page > 0:
        nav_buttons.append(InlineKeyboardButton(text="⬅️ Назад", callback_data=f"display_receipts_page_{page-1}"))
    if end_index < len(receipts):
        nav_buttons.append(InlineKeyboardButton(text="Вперед ➡️", callback_data=f"display_receipts_page_{page+1}"))

    if nav_buttons:
        keyboard.append(nav_buttons)

    keyboard.append([InlineKeyboardButton(text="К истории покупок", callback_data="back_to_history_delete")])

    markup = InlineKeyboardMarkup(inline_keyboard=keyboard)

    if message.message_id:
        await bot.edit_message_text(
            chat_id=user_id,
            message_id=message.message_id,
            text="Выберите чек для просмотра:",
            reply_markup=markup
        )
    else:
        await bot.send_message(
            chat_id=user_id,
            text="Выберите чек для просмотра:",
            reply_markup=markup
        )

    return markup

@dp.callback_query(lambda c: c.data and c.data == "history_receipts")
async def handle_history_receipts(callback_query: CallbackQuery):
    """
    Handles the callback query with the "history_receipts" data.

    Acknowledges the callback to prevent loading icons,
    removes the menu message and options,
    and then shows the user's receipts history with pagination.

    :param callback_query: The callback query containing the user ID
    """
    await callback_query.answer()  # Acknowledge the callback to prevent loading icons
    await bot.delete_message(chat_id=callback_query.message.chat.id, message_id=callback_query.message.message_id)

    user_id = callback_query.from_user.id

    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT *
            FROM user_purchases
            WHERE user_id = ?
        """, (user_id,))
        receipts = cursor.fetchall()

    if not receipts:
        back_button = create_back_button(text="Назад", callback_data="back_to_history_delete")
        await bot.send_message(user_id, "Ваша история покупок пока пуста", reply_markup=back_button)
        return

    sorted_receipts = get_sorted_user_receipts(user_id)

    # Create a new message if there isn't an existing one to edit
    if callback_query.message.text != "Выберите чек для просмотра:":
        new_message = await bot.send_message(user_id, "Загрузка истории покупок...")
    else:
        new_message = callback_query.message

    await display_receipts(user_id, new_message, sorted_receipts)


# ======= Receipt display =======

@dp.callback_query(lambda c: c.data and c.data.startswith("display_receipts"))
async def handle_display_receipts(callback_query: CallbackQuery):
    """
    Handles the callback query with the "display_receipts" data.

    Acknowledges the callback to prevent loading icons,
    removes the menu message and options,
    and then shows the user's receipts history with pagination or a specific receipt.

    :param callback_query: The callback query containing the user ID and the page or receipt index
    """
    await callback_query.answer()  # Acknowledge the callback to prevent loading icons

    if callback_query.data.startswith("display_receipts_page_"):
        page = int(callback_query.data[len("display_receipts_page_"):])
        user_id = callback_query.from_user.id
        sorted_receipts = get_sorted_user_receipts(user_id)
        await display_receipts(user_id, callback_query.message, sorted_receipts, page)
        return
    elif callback_query.data.startswith("display_receipts_receipt_"):
        await bot.delete_message(chat_id=callback_query.message.chat.id, message_id=callback_query.message.message_id)
        user_id = callback_query.from_user.id
        receipt_id = callback_query.data[len("display_receipts_receipt_"):].strip()
        await display_single_receipt(user_id, receipt_id)
        return
    
    await show_history_options(callback_query.message) #? What is this for? Remove?

async def display_single_receipt(user_id: int, receipt_id: str):
    """
    Displays a single receipt to the user.

    Args:
    user_id (int): The user ID to send the message to.
    receipt_id (str): The receipt ID to display.

    Returns:
    None
    """
    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT *
            FROM receipt_items
            WHERE receipt_id = ?
        """, (receipt_id,))
        receipt_items = cursor.fetchall()

        cursor.execute("""
            SELECT *
            FROM user_purchases
            WHERE receipt_id = ?
        """, (receipt_id,))
        receipt = cursor.fetchone()

    items = [f'{i}. "{item["name"]}" - {item["quantity"]} - {item["user_prediction"]} ({int(item["confidence"]*100)})%' for i, item in enumerate(receipt_items, start=1)]
    message = (f'{receipt["purchase_datetime"]} - {receipt["total_sum"]}₽\n'
               'Продукты:\n\n')
    message += '\n'.join(items)
    back_button = create_back_button(text="Назад", callback_data="history_receipts")
    await bot.send_message(user_id, message, reply_markup=back_button)
    

# ======= Categories display =======
user_messages_categories = defaultdict(list)

async def display_categories(user_id, message, product_counts, undetected_categories, page=0):
    """
    Displays a list of categories to the user, with the total amount of each category.
    The function takes a page parameter to split the list into pages.
    Each category is a button that, when pressed, shows a list of sources for that category.
    The function also shows navigation buttons to move between pages and a button to go back to the history.
    """
    keyboard = []
    categories_list = list(product_counts.items()) + list(undetected_categories.items())
    start_index = page * 7
    end_index = min(start_index + 7, len(categories_list))

    for i in range(start_index, end_index):
        category, data = categories_list[i]
        total_amount = data['total_amount']
        if total_amount is None:
            display_text = f"{category}: не определено"
        else:
            display_text = f"{category}: {total_amount:.2f} г"
        keyboard.append([InlineKeyboardButton(text=display_text, callback_data=f"display_categories_category_{i}")])

    nav_buttons = []
    if page > 0:
        nav_buttons.append(InlineKeyboardButton(text="⬅️ Назад", callback_data=f"display_categories_page_{page-1}"))
    if end_index < len(categories_list):
        nav_buttons.append(InlineKeyboardButton(text="Вперед ➡️", callback_data=f"display_categories_page_{page+1}"))

    if nav_buttons:
        keyboard.append(nav_buttons)

    keyboard.append([InlineKeyboardButton(text="К истории покупок", callback_data="back_to_history_delete")])

    markup = InlineKeyboardMarkup(inline_keyboard=keyboard)

    if message.message_id:
        await bot.edit_message_text(
            chat_id=user_id,
            message_id=message.message_id,
            text="Выберите категорию для просмотра источников:",
            reply_markup=markup
        )
    else:
        sent_message = await bot.send_message(
            chat_id=user_id,
            text="Выберите категорию для просмотра источников:",
            reply_markup=markup
        )
        user_messages_categories[user_id] = [sent_message.message_id]

    return markup

@dp.callback_query(lambda c: c.data and c.data.startswith("history_categories"))
async def handle_display_categories(callback_query: CallbackQuery):
    """
    Handles the callback query with the "history_categories" data.

    Acknowledges the callback to prevent loading icons,
    removes the menu message and options,
    and then shows the user's categories with their total amounts.
    If the user has no purchases, it shows a message saying so.
    The function also shows navigation buttons to move between pages and a button to go back to the history.

    :param callback_query: The callback query containing the user ID
    """
    await callback_query.answer()
    user_id = callback_query.from_user.id
    await bot.delete_message(chat_id=callback_query.message.chat.id, message_id=callback_query.message.message_id)

    # Delete all previous messages
    for message_id in user_messages_categories[user_id]:
        try:
            await bot.delete_message(chat_id=user_id, message_id=message_id)
        except Exception:
            pass  # Message may have already been deleted
    user_messages_categories[user_id] = []

    product_counts, undetected_categories = count_product_amounts(user_id)

    if not product_counts and not undetected_categories:
        back_button = create_back_button(text="К истории покупок", callback_data="back_to_history_delete")
        sent_message = await bot.send_message(user_id, "Ваша история покупок пока пуста", reply_markup=back_button)
        user_messages_categories[user_id] = [sent_message.message_id]
        return

    new_message = await bot.send_message(user_id, "Загрузка данных о категориях продуктов...")
    user_messages_categories[user_id] = [new_message.message_id]

    await display_categories(user_id, new_message, product_counts, undetected_categories)

@dp.callback_query(lambda c: c.data and c.data.startswith("display_categories_page_"))
async def handle_categories_page(callback_query: CallbackQuery):
    """
    Handles the callback query with the "display_categories_page_" data.

    Acknowledges the callback to prevent loading icons,
    removes the menu message and options,
    and then shows the user's categories with their total amounts
    on the specified page.
    If the user has no purchases, it shows a message saying so.
    The function also shows navigation buttons to move between pages and a button to go back to the history.

    :param callback_query: The callback query containing the user ID and the page number
    """
    await callback_query.answer()
    page = int(callback_query.data.split("_")[-1])
    user_id = callback_query.from_user.id
    
    product_counts, undetected_categories = count_product_amounts(user_id)
    await display_categories(user_id, callback_query.message, product_counts, undetected_categories, page)

@dp.callback_query(lambda c: c.data and c.data.startswith("display_categories_category_"))
async def handle_category_sources(callback_query: CallbackQuery):
    """
    Handles the callback query with the "display_categories_category_" data.

    Acknowledges the callback to prevent loading icons,
    removes the menu message and options,
    and then shows the user's sources for the category with the specified index.
    If the category has an undefined total amount, it shows a list of sources.
    Otherwise, it shows the total amount of the category and the sources with their amounts.
    The function also shows a button to go back to the categories list.

    :param callback_query: The callback query containing the user ID and the category index
    """
    await callback_query.answer()
    category_index = int(callback_query.data.split("_")[-1])
    user_id = callback_query.from_user.id
    
    # Delete all previous messages
    for message_id in user_messages_categories[user_id]:
        try:
            await bot.delete_message(chat_id=user_id, message_id=message_id)
        except Exception:
            pass  # Message may have already been deleted
    user_messages_categories[user_id] = []

    product_counts, undetected_categories = count_product_amounts(user_id)
    categories_list = list(product_counts.items()) + list(undetected_categories.items())
    category, data = categories_list[category_index]
    
    if data['total_amount'] is None:
        sources_text = f"Категория: {category}\n\nНеопределенные продукты:\n\n"
        for i, source in enumerate(data['sources'], start=1):
            sources_text += f"{i}. {source}\n"
    else:
        sources_text = f"Категория: {category}\nОбщее количество: {data['total_amount']:.2f} г\n\nИсточники:\n\n"
        for i, (source, amount) in enumerate(data['sources'], start=1):
            sources_text += f"{i}. {source}: {amount:.2f} г\n"
    
    back_button = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Назад к категориям", callback_data="history_categories")]
    ])
    
    if len(sources_text) < 4096:
        sent_message = await bot.send_message(
            chat_id=user_id,
            text=sources_text,
            reply_markup=back_button
        )
        user_messages_categories[user_id] = [sent_message.message_id]
    else:
        chunks = await split_message_into_chunks(sources_text)
        for i, chunk in enumerate(chunks):
            if i == len(chunks) - 1:
                sent_message = await bot.send_message(
                    chat_id=user_id,
                    text=chunk,
                    reply_markup=back_button
                )
            else:
                sent_message = await bot.send_message(
                    chat_id=user_id,
                    text=chunk,
                )
            user_messages_categories[user_id].append(sent_message.message_id)

async def split_message_into_chunks(text, chunk_size=4096):
    """
    Split a given text into chunks of maximum size `chunk_size` (default 4096)
    to be used in Telegram messages.

    :param text: The text to be split
    :param chunk_size: The maximum size of a chunk
    :return: A list of strings, each of length `chunk_size` or less
    """
    chunks = []
    current_chunk = ""
    for line in text.split("\n"):
        if len(current_chunk) + len(line) + 1 > chunk_size:
            chunks.append(current_chunk)
            current_chunk = line
        else:
            current_chunk += line + "\n"
    if current_chunk:
        chunks.append(current_chunk[:-1])
    return chunks


# ======= Microelements display =======

async def display_microelements(user_id, message, microelements_data, page=0):
    """
    Displays a list of microelements to the user, with the total amount of each microelement.
    The function takes a page parameter to split the list into pages.
    Each microelement is a button that, when pressed, shows a list of sources for that microelement.
    The function also shows navigation buttons to move between pages and a button to go back to the history.

    :param user_id: The user ID to send the message to
    :param message: The message object to edit or send
    :param microelements_data: A dictionary with microelements as keys and dictionaries with total_amount and sources as values
    :param page: The page number to show
    :return: The InlineKeyboardMarkup object with the buttons
    """
    keyboard = []
    microelements_list = list(microelements_data.items())
    start_index = page * 7
    end_index = min(start_index + 7, len(microelements_list))

    for i in range(start_index, end_index):
        element, data = microelements_list[i]
        total_amount = data['total_amount']
        if element == ABBREVIATION_MICROELEMENTS_DICT["эц"]:
            unit = 'Ккал'
        else:
            unit = 'г'
        keyboard.append([InlineKeyboardButton(text=f"{element}: {total_amount:.2f} {unit}", callback_data=f"display_microelements_element_{i}")])

    nav_buttons = []
    if page > 0:
        nav_buttons.append(InlineKeyboardButton(text="⬅️ Назад", callback_data=f"display_microelements_page_{page-1}"))
    if end_index < len(microelements_list):
        nav_buttons.append(InlineKeyboardButton(text="Вперед ➡️", callback_data=f"display_microelements_page_{page+1}"))

    if nav_buttons:
        keyboard.append(nav_buttons)

    keyboard.append([InlineKeyboardButton(text="К истории покупок", callback_data="back_to_history_delete")])

    markup = InlineKeyboardMarkup(inline_keyboard=keyboard)

    if message.message_id:
        await bot.edit_message_text(
            chat_id=user_id,
            message_id=message.message_id,
            text="Выберите микроэлемент для просмотра источников:",
            reply_markup=markup
        )
    else:
        await bot.send_message(
            chat_id=user_id,
            text="Выберите микроэлемент для просмотра источников:",
            reply_markup=markup
        )

    return markup

@dp.callback_query(lambda c: c.data and c.data.startswith("history_microelements"))
async def handle_display_microelements(callback_query: CallbackQuery):
    """
    Handles the callback query with the "history_microelements" data.

    Acknowledges the callback to prevent loading icons,
    removes the menu message and options,
    and then shows the user's microelements data with pagination.
    If the user has no purchases, it shows a message saying so.
    The function also shows navigation buttons to move between pages and a button to go back to the history.

    :param callback_query: The callback query containing the user ID
    """
    await callback_query.answer()  # Acknowledge the callback to prevent loading icons
    await bot.delete_message(chat_id=callback_query.message.chat.id, message_id=callback_query.message.message_id)

    user_id = callback_query.from_user.id
    
    microelements_data = get_microelements_data(user_id, MICROELEMENTS_TABLE)

    if not microelements_data:
        back_button = create_back_button(text="Назад", callback_data="back_to_history_delete")
        await bot.send_message(user_id, "Данные о микроэлементах отсутствуют", reply_markup=back_button)
        return

    # Create a new message if there isn't an existing one to edit
    new_message = await bot.send_message(user_id, "Загрузка данных о микроэлементах...")

    await display_microelements(user_id, new_message, microelements_data)

@dp.callback_query(lambda c: c.data and c.data.startswith("display_microelements_page_"))
async def handle_microelements_page(callback_query: CallbackQuery):
    """
    Handles the callback query with the "display_microelements_page_" data.

    Acknowledges the callback to prevent loading icons,
    removes the menu message and options,
    and then shows the user's microelements data on the specified page.
    If the user has no purchases, it shows a message saying so.
    The function also shows navigation buttons to move between pages and a button to go back to the history.

    :param callback_query: The callback query containing the user ID and the page number
    """
    await callback_query.answer()
    page = int(callback_query.data.split("_")[-1])
    user_id = callback_query.from_user.id
    
    microelements_data = get_microelements_data(user_id, MICROELEMENTS_TABLE)
    await display_microelements(user_id, callback_query.message, microelements_data, page)

@dp.callback_query(lambda c: c.data and c.data.startswith("display_microelements_element_"))
async def handle_microelement_sources(callback_query: CallbackQuery):
    """
    Handles the callback query with the "display_microelements_element_" data.

    Acknowledges the callback to prevent loading icons,
    removes the menu message and options,
    and then shows the user's sources for the specified microelement.
    The function also shows a button to go back to the microelements list.

    :param callback_query: The callback query containing the user ID and the microelement index
    """
    await callback_query.answer()
    element_index = int(callback_query.data.split("_")[-1])
    user_id = callback_query.from_user.id
    
    microelements_data = get_microelements_data(user_id, MICROELEMENTS_TABLE)
    element, data = list(microelements_data.items())[element_index]
    if element == ABBREVIATION_MICROELEMENTS_DICT["эц"]:
        unit = 'Ккал'
    else:
        unit = 'г'
    sources_text =(f"Общее количество: {data['total_amount']:.2f} {unit}\n"
                    f"Источники {element}:\n\n")
    for i, (source, amount) in enumerate(data['sources'], start=1):
        sources_text += f"{i}. {source}: {amount:.6f} {unit}\n"
    
    back_button = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Назад к микроэлементам", callback_data="history_microelements")]
    ])
    if len(sources_text) < 4096:
        await bot.edit_message_text(
            chat_id=user_id,
            message_id=callback_query.message.message_id,
            text=sources_text,
            reply_markup=back_button
        )
    else:
        chunks = await split_message_into_chunks(sources_text, chunk_size=4096)
        for i, chunk in enumerate(chunks):
            if i == 0:
                await bot.edit_message_text(
                    chat_id=user_id,
                    message_id=callback_query.message.message_id,
                    text=chunk,
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
                    text=chunk,
                )

# ======= Main loop =======
async def main():
    """
    Main entry point of the bot. Starts the bot and waits for new messages.
    The bot will start polling for new messages and will stop when an exception is raised.
    When the bot is stopped, it will close the aiohttp session.
    """
    try:
        make_backup()
        print("Starting bot...")
        await dp.start_polling(bot)  
    finally:
        make_backup()
        print("Stopping bot...")
        await bot.session.close()  


if __name__ == "__main__":
    asyncio.run(main())