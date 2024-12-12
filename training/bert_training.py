import tensorflow as tf
from model.bert_model import BertModel
import pandas as pd
import datetime
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from training.test_utils import model_evaluate_during_training, plot_training_history, find_best_epoch
from config.config import DATASETS_FOLDER, MODEL_CONFIG_PATH, TRAINING_HISTORY_PATH
import json
import shutil

"""
Этот код реализует процесс обучения модели BERT на основе парсированных данных. 
Он включает следующие этапы:
1. Загрузка и предобработка данных.
2. Инициализация модели BERT.
3. Обучение модели с использованием обратных вызовов для сохранения весов и ранней остановки.
4. Тестирование модели и визуализация истории обучения.

Основные компоненты:
- `CustomSaver`: Класс для сохранения весов модели на определённой частоте.
- `BertModel`: Модель для классификации текстовых данных.
- `train_test_split`: Разделение данных на обучающую и тестовую выборки.
- `model_evaluate_during_training` и `plot_training_history`: Функции для тестирования модели и визуализации.

Ключевые переменные:
- `data_path`: Путь к CSV-файлу с входными данными.
- `NUM_CLASSES`: Количество уникальных классов в метках.
- `callbacks`: Список обратных вызовов для управления процессом обучения.
"""


class CustomSaver(tf.keras.callbacks.Callback):
    """
    Класс обратного вызова для сохранения весов модели после каждой эпохи.
    """
    def __init__(self, save_freq=1, folder_name='model_weights'):
        """
        Инициализация объекта CustomSaver.

        Параметры:
        - save_freq (int): Частота сохранения (каждые `save_freq` эпох).
        - folder_name (str): Папка для сохранения весов.
        """
        super(CustomSaver, self).__init__()
        self.save_freq = save_freq
        self.folder_name = folder_name
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)
            print(f"Создана директория: {self.folder_name}")

    def on_epoch_end(self, epoch, logs=None):
        """
        Сохранение весов модели после завершения эпохи.

        Параметры:
        - epoch (int): Номер эпохи.
        - logs (dict): Логи текущей эпохи, содержащие метрики.
        """
        if (epoch + 1) % self.save_freq == 0:
            accuracy = logs.get('accuracy')
            loss = logs.get('loss')
            if accuracy is not None and loss is not None:
                filename = f"model_{accuracy:.4f}_{loss:.4f}_{epoch+1}.h5"
                filepath = os.path.join(self.folder_name, filename)
                self.model.save_weights(filepath)
                print(f"\nВеса модели сохранены в: {filepath}")
            else:
                print("\nМетрики accuracy или loss недоступны.")


# Загрузка конфигурации модели
with open(MODEL_CONFIG_PATH, 'r') as f:
    MODEL_CONFIG = json.load(f)

# Параметры обучения
DATA_PATH = os.path.join(DATASETS_FOLDER, 'data_nov28_combined_parsed.csv')
NUM_EPOCHS = 60  # Кол-во эпох тренировки
TEST_SPLIT = 0.1  # Сколько данных отделить на тестирование модели
BATCH_SIZE = 80   # Сколько примеров модель будет обрабатывать параллельно
SAVING_WEIGHTS_FREQUENCY = 1
EARLY_STOPPING_PATIENECE = 8  # Количество эпох для ранней остановки

# Загрузка и подготовка данных
df = pd.read_csv(DATA_PATH, sep='|').fillna('empty')
df['clear_name'] = df['clear_name'].apply(lambda x: x.lower())
df['product_name'] = df['product_name'].apply(lambda x: x.lower())
# Исключаем класс "несъедобное"
df = df[df['clear_name'] != 'несъедобное']

print(f"Количество записей после фильтрации: {len(df)}")

x = df['product_name'].tolist()
y = df['clear_name'].tolist()

# Кодирование меток
le = LabelEncoder()
le.fit(y)
y = le.transform(y)
NUM_CLASSES = len(le.classes_)
print(f"Количество классов: {NUM_CLASSES}")

# Дублирование записей, если в классе только один пример
y_list = list(y)
single_example_classes = [label for label in set(y_list) if y_list.count(label) == 1]
for label in single_example_classes:
    index = y_list.index(label)
    x.append(x[index])
    y_list.append(y_list[index])
y = y_list

# Разделение данных на обучающую и тестовую выборки
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=TEST_SPLIT,
    random_state=42,
    stratify=y
)

# Инициализация модели
bert_model = BertModel(num_labels=NUM_CLASSES, max_length=MODEL_CONFIG['max_length'])
model_full_name = f'bert_multilabel_weights_{MODEL_CONFIG["model_name"]}_{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")}'

# Настройка путей сохранения
saving_path = os.path.join(TRAINING_HISTORY_PATH, model_full_name)

# Обратные вызовы
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENECE)
callbacks = [CustomSaver(save_freq=SAVING_WEIGHTS_FREQUENCY, folder_name=saving_path), es]

# Обучение модели
history = bert_model.train(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')],
    callbacks=callbacks
)

# Тестирование модели
print('Тестирование модели...')
history_train, history_test = model_evaluate_during_training(
    bert_model,
    saving_path,
    train_data=(x_train, y_train),
    test_data=(x_test, y_test),
    save_history=True
)

# Визуализация истории обучения
plot_training_history(
    history_train,
    history_test,
    path_to_json=None,
    saving_path=os.path.join(saving_path, 'history_plot.png')
)

# Сохраняем лучшие веса отдельно
json_filename = [file for file in os.listdir(saving_path) if file.endswith('.json')][0]
best_epoch = find_best_epoch(os.path.join(saving_path, json_filename), deciding_set='test') + 1

for filename in os.listdir(saving_path):
    epoch = filename.split('.h')[0].split('_')[-1]
    if str(epoch) == str(best_epoch):
        best_weights_filename = filename

best_weights_path = os.path.join(saving_path, best_weights_filename)
shutil.copy(best_weights_path, os.path.join(MODEL_CONFIG['weights_path'], model_full_name, '.h5'))
