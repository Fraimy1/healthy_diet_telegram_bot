from pydantic import BaseModel
from openai import OpenAI
import openai
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import time

"""
Этот модуль предназначен для генерации синтетических данных с использованием OpenAI API. 
Основные возможности:
1. Генерация новых записей для указанного класса, обеспечивая их уникальность.
2. Оверсэмплинг набора данных, чтобы в каждом классе было не менее заданного количества записей.
3. Использование примеров из существующих данных для улучшения генерации.

Ключевые компоненты:
- `generate_entries`: Генерация записей для указанного класса.
- `generate_class_entries`: Создание записей для отдельного класса с использованием примеров.
- `oversample_dataset`: Увеличение числа записей для недостаточно заполненных классов в наборе данных.

Переменные:
- `prompt`: Запрос для OpenAI, описывающий задачу генерации данных.

Результат:
- Обновлённый набор данных с увеличенным количеством записей для каждого класса.
"""

prompt = '''Your task is to create UNIQUE new data to a CERTAIN CLASS which will be specified. You will be provided with some already used data which you should NEVER reuse and a number of examples you must create. You are free to come up with new brand names or use already existent you know. Your main goal is to make a required number of new entries which must be UNIQUE to those already existing, this data can be dirty like in the examples provided, like the one you'd see in receipt. This data is needed to train an AI classification model.
```
YOUR DATA SHOULD BE IN RUSSIAN LANGUAGE!
YOU SHOULD GENERATE EXACTLY AS MANY ENTRIES AS USER ASKS IN <number_entries_to_generate>!
```
Examples:
<user_input>
<class_name>пиво светлое, с долей сухих в-в в исходном сусле 11% 11.2.1.1
<number_entries_to_generate> 1
<available_data> ['пиво сибирская корона классическое светлое паст. 0.47л с/б','пиво "губернское живое" 1,5л','0,45л пиво охота крепкое ст', 'пиво чешский медведь живое 1,35л пэт/балтика']
</user_input>
<model_output>['пиво бархатное тёмное 3,7%']</model_output>
```
<user_input>
<class_name>хлеб бородинский (мука ржаная обойная и пшеничная 2 сорта) 6.2.2.1
<available_data> ['хлеб монастырский двор российский бородинский 300г', 'хлеб бородинский 0,4 уп нарезка', 'хлеб рж-пшен бор 300']
<number_entries_to_generate> 2
</user_input>
<model_output>['хлеб бородино( с кориандром) 250 г', 'хлебушек бородинский']</model_output>'''

class GeneratedEntry(BaseModel):
    """
    Модель данных для отдельной сгенерированной записи.
    """
    entry: str
    entry_number: int

class GeneratedEntries(BaseModel):
    """
    Модель данных для списка сгенерированных записей.
    """
    entries: list[GeneratedEntry]

def generate_entries(
    class_name: str, 
    client: OpenAI, 
    num_of_new_entries: int = 1, 
    available_data: list = []
) -> list:
    """
    Генерирует новые записи для указанного класса с использованием OpenAI API.

    Аргументы:
    - class_name (str): Имя класса для генерации данных.
    - client (OpenAI): Клиент OpenAI для генерации данных.
    - num_of_new_entries (int): Количество новых записей для генерации.
    - available_data (list): Список существующих данных для примера.

    Возвращает:
    - list: Список сгенерированных записей.
    """
    assert num_of_new_entries > 0, "Количество новых записей должно быть больше 0"
    # Создание входной строки для OpenAI API
    input = f'''<class_name> {class_name}
    <available_data> {available_data}
    <number_entries_to_generate> {num_of_new_entries}'''

    try:
        # Генерация данных с использованием OpenAI API
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": f"{prompt}"},
                {"role": "user", "content": f"{input}"},
            ],
            response_format=GeneratedEntries,
        )
    except Exception as e:
        if type(e) == openai.LengthFinishReasonError:
            print("Слишком много токенов: ", e)
            pass
        else:
            print(e)
            pass

    # Извлечение сгенерированных данных из ответа
    generated_data = [entry.entry for entry in completion.choices[0].message.parsed.entries]
    return generated_data

def generate_class_entries(
    class_name: str,
    client: OpenAI,
    num_entries_to_generate: int,
    dataset: pd.DataFrame,
    num_example_entries: int = -1
) -> list:
    """
    Генерирует новые записи для указанного класса.

    Аргументы:
    - class_name (str): Имя класса для генерации данных.
    - client (OpenAI): Клиент OpenAI.
    - num_entries_to_generate (int): Количество новых записей.
    - dataset (pd.DataFrame): Набор данных с существующими примерами.
    - num_example_entries (int): Количество примеров для генерации. (-1 - использовать все доступные).

    Возвращает:
    - list: Список сгенерированных записей.
    """
    # Фильтрация записей для указанного класса
    class_entries = dataset[dataset['clear_name'] == class_name]['name']
    if num_example_entries == -1:
        num_example_entries = len(class_entries)

    # Определение примеров для генерации
    example_entries = np.random.choice(
        class_entries,
        min(num_example_entries, len(class_entries)),
        replace=False
    )

    # Генерация данных
    generated_entries = generate_entries(
        class_name,
        client,
        num_entries_to_generate,
        example_entries
    )
    return generated_entries

def oversample_dataset(data, client, min_entries, example_count=-1, checkpoint_frequency=None) -> pd.DataFrame:
    """
    Осуществляет оверсэмплинг набора данных для достижения минимального количества записей в каждом классе.

    Аргументы:
    - data (pd.DataFrame): Набор данных с колонками 'clear_name' (класс) и 'name' (запись).
    - client (OpenAI): Клиент OpenAI для генерации данных.
    - min_entries (int): Минимальное количество записей для каждого класса.
    - example_count (int, optional): Количество примеров для генерации (-1 - использовать все доступные).
    - checkpoint_frequency (int, optional): Частота сохранения контрольных точек.

    Возвращает:
    - pd.DataFrame: Обновлённый набор данных с оверсэмплингом.
    """
    class_names = data['clear_name'].unique()
    total_classes = len(class_names)
    print(f"Оверсэмплинг набора данных. Всего классов: {total_classes}")
    
    for idx, name in enumerate(class_names):
        print('-'*50)
        print(f"Оверсэмплинг класса {idx + 1}/{total_classes}: {name}")
        entries = data[data['clear_name'] == name]['name']
        
        if len(entries) < min_entries:
            print(f"Класс {name} содержит {len(entries)} записей, что меньше {min_entries}. Выполняется оверсэмплинг...")
            start_time = time.time()
            
            entries_needed = min_entries - len(entries)
            
            while entries_needed > 10:
                try:
                    new_entries = generate_class_entries(name, client, min_entries, data, example_count)
                    new_data = pd.DataFrame({'name': new_entries, 'clear_name': [name] * len(new_entries)})
                    data = pd.concat([data, new_data], ignore_index=True)
                    entries_needed -= len(new_entries)
                    if checkpoint_frequency and idx % checkpoint_frequency == 0:
                        data.to_csv(f'data_checkpoint_{idx % checkpoint_frequency}.csv', index=False)
                except Exception as e:
                    print('ОШИБКА!', e)

                if entries_needed > 0:
                    print(f"Сгенерировано {min_entries - entries_needed} новых записей. Продолжается генерация...")

            elapsed_time = time.time() - start_time
            print(f"\033[36mКласс {name} обработан за {elapsed_time:.2f} секунд\033[0m")
        else:
            print(f"Класс {name} содержит достаточно записей ({len(entries)}). Пропускается...")
    
    print("Оверсэмплинг завершён")
    return data
