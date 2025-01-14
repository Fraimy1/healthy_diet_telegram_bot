"""
Этот скрипт загружает данные из CSV файла, очищает названия продуктов, объединяет данные и сохраняет результат в новый CSV файл.
"""

from utils.parser import Parser
import pandas as pd
import os
from config.config import DATASETS_FOLDER

FILE_NAME = 'data_jan10_combined.csv'
PARSED_FILE_NAME = 'data_jan10_combined_parsed.csv'

# Загрузка данных из CSV файла
data = pd.read_csv(os.path.join(DATASETS_FOLDER, FILE_NAME), sep='|')

parser = Parser(verbose=False)

clean_data = parser.parse_dataset(data['name'], extract_hierarchical_number=False) # Парсинг
clean_data['original_name'] = data['name']
clean_data['clear_name'] = data['clear_name']
clean_data['product_name'] = parser.clean_dataset(clean_data['product_name'], 
                                                  remove_numbers=True, replace_punctuation_with_spaces=True) # Очистка названий

clean_data = clean_data[['original_entry', 'percentage', 'amount',
                          'product_name', 'original_name', 'clear_name', 'portion']] # Изменение порядка столбцов

clean_data.to_csv(os.path.join(DATASETS_FOLDER, PARSED_FILE_NAME), sep='|', index=False)