"""
Этот скрипт скачивает данные из Google Sheets, объединяет их и сохраняет в новый CSV файл.
"""

import pandas as pd
import gdown
import os
from config.repository_config import UNPROCESSED_DATASET_LINKS
from config.config import DATASETS_FOLDER, DATABASE_PATH

# Ссылка на скачивание листов из Google Sheets
edible_link = UNPROCESSED_DATASET_LINKS['edible']
inedible_link = UNPROCESSED_DATASET_LINKS['inedible']
microelements_link = UNPROCESSED_DATASET_LINKS['microelements']

# Пути для сохранения скачанных CSV файлов
data_path_edible = os.path.join(DATASETS_FOLDER, 'data_jan10_edible.csv')
data_path_inedible = os.path.join(DATASETS_FOLDER, 'data_jan10_inedible.csv')
data_path_microelements = os.path.join(DATABASE_PATH, 'jan10_microelements_dirty.xlsx')

# Скачивание листов из Google Sheets
print("Скачивание съедобных...")
gdown.download(edible_link, data_path_edible, quiet=False)

print("Скачивание несъеднобных...")
gdown.download(inedible_link, data_path_inedible, quiet=False)

print("Скачивание микроэлементов/ТХС...")
gdown.download(microelements_link, data_path_microelements, quiet=False)

# Названия столбцов для CSV файлов
column_names = [
    'name',
    'clear_name',
    'product_code',
    'closest_category',
    'closest_category_code',
    'additional_category1',
    'additional_category2'
]

# Чтение CSV файлов в pandas DataFrames
print("Чтение листа 1 в DataFrame...")
data_sheet_edible = pd.read_csv(data_path_edible, sep=',', usecols=range(5))  # Take first 5 columns
data_sheet_edible.columns = column_names[:5]  # Rename columns to match first 5 names

print("Чтение листа 2 в DataFrame...")
data_sheet_inedible = pd.read_csv(data_path_inedible, sep=',', usecols=range(2))  # Take first 2 columns
data_sheet_inedible.columns = column_names[:2]  # Rename columns to match first 2 names

# Объединение двух DataFrames
print("Объединение двух DataFrames...")
combined_data = pd.concat([data_sheet_edible, data_sheet_inedible], ignore_index=True)

# Удаление строк с пропущенными значениями в столбце 'name'
print("Удаление строк с пропущенными значениями в столбце 'name'...")
combined_data.dropna(subset=['name'], inplace=True)

# Сохранение объединенного DataFrame в новый CSV файл
output_path = os.path.join(DATASETS_FOLDER, 'data_jan10_combined.csv')
print(f"Сохранение объединенных данных в {output_path}...")
combined_data.to_csv(output_path, sep='|', index=False)

# Вывод DataFrame и его информации
print("\nОбъединенные данные:")
print(combined_data)

print("\nИнформация о DataFrame:")
print(combined_data.info())