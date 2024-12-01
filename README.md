- [README in English](#diet-analysis-system-for-russian-receipts-with-telegram-bot-integration)
- [README на русском](#система-анализа-диеты-на-основе-чеков-с-интеграцией-telegram-бота)

# Diet Analysis System for Russian Receipts with Telegram Bot Integration

The system classifies product names from receipts sent via a Telegram bot and builds a user's diet history.

## Table of Contents

- [Project Description](#project-description)
- [Project Specifics](#project-specifics)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Directory Structure](#directory-structure)

## Project Description
This project addresses a common issue with diet tracking apps: the time and effort required to log daily food intake, which often leads to users abandoning the habit. By integrating a Telegram bot and an AI classification algorithm, users can easily add products to their diet history by simply scanning a QR code on a receipt.

Users can then view the products they purchased, how they were classified by the algorithm, and detailed nutritional information (calories, microelements, etc.).

In the future, a scoring system will be implemented, giving users a score (1-100) to show how their diet compares to an ideal diet in their region.

## Technologies used:
- Aiogram
- BERT

## Project Specifics
Due to an NDA with the non-commercial company this project is being developed for, I cannot share the following:
- **Training data** for the BERT model
- **Model weights** used in the BERT model
- **Microelements table** provided by the company

If you have access to this data, you should specify it on step 4 of the installation instruction. 
## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/Fraimy1/healthy_diet_telegram_bot.git
    ```

2. Navigate to the project directory (replace `path_to_project_folder` to an actual path):

    ```bash
    cd path_to_project_folder
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Set up the configuration by editing `config/repository_config.py` adding the links to Google drive with the data and model weights.
     Example (Replace `VALID_ID` with actual ids provided to you by the developer): 
     ```python
    # Links to model weights

    LABEL_ENCODER_LINK = 'https://drive.google.com/file/d/VALID_ID/view?usp=drive_link'
    BERT_MODEL_LINK = 'https://drive.google.com/file/d/VALID_ID/view?usp=drive_link'
    BINARY_MODEL_LINK = 'https://drive.google.com/file/d/VALID_ID/view?usp=drive_link'
    
    # Links to data
    
    DYNAMIC_DATASET_LINK = 'https://drive.google.com/file/d/VALID_ID/view?usp=drive_link'
    CLEANED_MICROELEMENTS_TABLE_LINK = 'https://drive.google.com/file/d/VALID_ID/view?usp=drive_link'
     ```

5. Run the data_init file to download necessary components and create necessary files (e.g., database, model_weights):

    ```bash
    python -m repository_initialization.data_init
    ```
6. In the created config/.env file replace the string with an actual telegram token:
     ```env
    TELEGRAM_TOKEN=0000000000:AAAAAAAA-BBBBBBBBBBBBB
     ```

## Usage

1. Start by initializing the necessary components (e.g., database, model_weights):

    ```bash
    python -m repository_initialization.data_init
    ```

2. Run the Telegram bot:

    ```bash
    python -m bot.telegram_bot
    ```
    
3. Optional: If you need to perform additional tasks like testing the model, refer to the relevant scripts in the `model` and `utils` directories.

## Configuration

The configuration files can be found in the `config/` directory. These files control the behavior of the project:

- `config.py`: Configuration for the bot (e.g., bot messages, instruction etc.)
- `repository_config.py`: Repository-related settings (e.g., model weights, data for training, microelements table).
- `model_config.json`: Configuration related to the model.

Ensure you update these files with your specific environment settings.

## Directory Structure

Here is a breakdown of the project’s directory structure excluding unneccessary files like `__init__.py`, `__pycache__` etc.<br>
├── README.md<br>
├── bot<br>
│   └── telegram_bot.py - main file for running the bot<br>
├── config<br>
│   ├── config.py<br>
│   ├── .env - is created after running `data_init.py`<br>
│   ├── model_config.json<br>
│   └── repository_config.py<br>
├── data<br>
│   ├── database - is created after running `data_init.py`<br>
│   └── database_init.py<br>
├── instruction_images<br>
│   ├── (1-9).jpg<br>
├── model<br>
│   ├── bert_model.py<br>
│   ├── model_init.py<br>
│   └── model_weights - is created after running `data_init.py`<br>
├── repository_initialization<br>
│   └── data_init.py - initialization of the project big files<br>
├── requirements.txt<br>
└── utils<br>
    ├── data_processor.py<br>
    ├── parser.py<br>
    └── utils.py<br>

# Система анализа диеты на основе чеков с интеграцией Telegram-бота

Система классифицирует наименования продуктов из чеков, отправленных через Telegram-бота, и формирует историю питания пользователя.

## Содержание

- [Описание проекта](#описание-проекта)
- [Особенности проекта](#особенности-проекта)
- [Установка](#установка)
- [Использование](#использование)
- [Конфигурация](#конфигурация)
- [Структура проекта](#структура-проекта)

## Описание проекта

Этот проект решает распространенную проблему в приложениях для отслеживания рациона: необходимость ежедневно записывать продукты, что требует много времени и усилий, и из-за этого пользователи часто бросают эту привычку. Интеграция с Telegram-ботом и алгоритмом ИИ для классификации позволяет пользователям автоматически добавлять продукты в свою историю питания, просто отсканировав QR-код на чеке.

Пользователи могут просматривать купленные продукты, видеть, как они были классифицированы алгоритмом, и получать подробную информацию о питательных веществах (калории, микроэлементы и т.д.).

В будущем будет реализована система оценки, которая будет показывать пользователю балл (от 1 до 100), отражающий, насколько здорово его питание по сравнению с идеальной диетой для его региона.

### Используемые технологии:
- Aiogram
- BERT

## Особенности проекта

Из-за соглашения о неразглашении с некоммерческой компанией, для которой разрабатывается этот проект, я **не могу предоставить** следующие данные:
- **Данные для обучения** модели BERT
- **Веса модели**, использующейся в BERT
- **Таблицу микроэлементов**, предоставленную компанией

Если у вас есть доступ к этим данным, вам нужно будет указать их на шаге 4 в инструкции по установке.

## Установка

1. Клонируйте репозиторий:

    ```bash
    git clone https://github.com/Fraimy1/healthy_diet_telegram_bot.git
    ```

2. Перейдите в директорию проекта (замените `path/to/project/folder` на реальный путь):

    ```bash
    cd path/to/project/folder
    ```

3. Установите необходимые зависимости:

    ```bash
    pip install -r requirements.txt
    ```

4. Настройте конфигурацию, отредактировав файл `config/repository_config.py`, добавив ссылки на Google Drive с данными и весами модели.
   Пример (замените `VALID_ID` на реальные ID, предоставленные вам разработчиком):

    ```python
    # Ссылки на веса модели

    LABEL_ENCODER_LINK = 'https://drive.google.com/file/d/VALID_ID/view?usp=drive_link'
    BERT_MODEL_LINK = 'https://drive.google.com/file/d/VALID_ID/view?usp=drive_link'
    BINARY_MODEL_LINK = 'https://drive.google.com/file/d/VALID_ID/view?usp=drive_link'
    
    # Ссылки на данные
    
    DYNAMIC_DATASET_LINK = 'https://drive.google.com/file/d/VALID_ID/view?usp=drive_link'
    CLEANED_MICROELEMENTS_TABLE_LINK = 'https://drive.google.com/file/d/VALID_ID/view?usp=drive_link'
    ```

5. Запустите файл `data_init`, чтобы загрузить необходимые компоненты и создать нужные файлы (например, базу данных, веса модели):

    ```bash
    python -m repository_initialization.data_init
    ```

6. В созданном файле `config/.env` вставьте реальный токен Telegram-бота:

    ```env
    TELEGRAM_TOKEN=0000000000:AAAAAAAA-BBBBBBBBBBBBB
    ```

## Использование

1. Начните с инициализации необходимых компонентов (например, базы данных, весов модели):

    ```bash
    python -m repository_initialization.data_init
    ```

2. Запустите Telegram-бота:

    ```bash
    python -m bot.telegram_bot
    ```

3. Опционально: если необходимо выполнить дополнительные задачи, такие как тестирование модели, обратитесь к соответствующим скриптам в папках `model` и `utils`.

## Конфигурация

Конфигурационные файлы находятся в папке `config/`. Эти файлы управляют поведением проекта:

- `config.py`: Конфигурация бота (сообщения, котоыре отправляет бот и т.п.)
- `repository_config.py`: Настройки, связанные с репозиторием (ссылки на веса модели, данные для обучения, таблицу микроэлементов).
- `model_config.json`: Настройки, связанные с моделью (пути к весам и т.п.).

Не забудьте обновить эти файлы, чтобы соответствовать вашему окружению.

## Структура проекта

Вот структура проекта, исключая лишние файлы вроде `__init__.py`, `__pycache__`<br>
├── README.md<br>
├── bot<br>
│   └── telegram_bot.py - основной файл для запуска бота<br>
├── config<br>
│   ├── config.py<br>
│   ├── .env - создается после запуска `data_init.py`<br>
│   ├── model_config.json<br>
│   └── repository_config.py<br>
├── data<br>
│   ├── database - создается после запуска `data_init.py`<br>
│   └── database_init.py<br>
├── instruction_images<br>
│   ├── (1-9).jpg<br>
├── model<br>
│   ├── bert_model.py<br>
│   ├── model_init.py<br>
│   └── model_weights - создается после запуска `data_init.py`<br>
├── repository_initialization<br>
│   └── data_init.py - инициализация больших файлов проекта<br>
├── requirements.txt<br>
└── utils<br>
    ├── data_processor.py<br>
    ├── parser.py<br>
    └── utils.py<br>
