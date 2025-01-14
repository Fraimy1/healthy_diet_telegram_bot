"""
Этот код выполняет обработку таблицы с данными о микроэлементах из CSV файла. 
Данные очищаются и преобразуются для дальнейшего анализа.
"""

import pandas as pd
import os
from config.config import DATABASE_PATH

# Чтение CSV файла
path = os.path.join(DATABASE_PATH, 'jan10_microelements_dirty.csv')
df = pd.read_csv(path, sep=',')

# Получение значений из первой строки
first_row = df.iloc[0]

# Группировка колонок по значениям в первой строке
grouped_dict = {}
for col in df.columns:
    key = first_row[col]
    if key is None or not type(key) == str:
        continue
    grouped_dict.setdefault(key, []).append(col)

print(grouped_dict, '-'*100, sep='\n')

# Удаление первой строки из DataFrame
df = df.drop(0).reset_index(drop=True)

# Удаление строк с пропусками в 'Продукт в ТХС'
df.dropna(subset=['Продукт в ТХС'], inplace=True)

# Удаление указанных колонок
df.drop(columns=['Степень обработки', "Пурины", "Оксалат", "Фруктоза", "Галактоза", "ТрансжирыУд.вес",
                 "Код продукта", "Ближайшая категория (БК)", "Код БК", "число продуктов  в датасете"],
        inplace=True, errors='ignore')

# Заполнение пропущенных значений нулями
df.fillna(0, inplace=True)

# Преобразование колонок в числовой формат (float)
for column in df.columns[1:]:

    df[column] = df[column].apply(lambda x: float(str(x).replace(',', '').replace(' ', '')) if isinstance(x, str) else float(x))
# Преобразование единиц измерения
unit_to_gram = {
    "гр": 1,
    "мг": 1e-3,
    "мкг": 1e-6
}

for unit, columns in grouped_dict.items():
    if unit == 'Ккал':
        continue
    for column in columns:
        df[column] = df[column].apply(lambda x: x * unit_to_gram[unit])

# Фильтрация DataFrame по наличию символа '|' в колонке 'Продукт в ТХС'
df_filtered = df[df['Продукт в ТХС'].str.contains('\|', regex=True)]

# Сохранение очищенных данных в формате CSV
output_path = os.path.join(DATABASE_PATH, 'cleaned_microelements_table.csv')
df.to_csv(output_path, index=False, sep='|')

# Проверка результата
df = pd.read_csv(output_path, sep='|')
print(df.head(4))
