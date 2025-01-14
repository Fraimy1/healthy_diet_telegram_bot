import tensorflow as tf
from model.bert_model import BertModel
import pandas as pd
import datetime
import os
import shutil  # Add missing import
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from training.test_utils import model_evaluate_during_training, plot_training_history, find_best_epoch
from config.config import DATASETS_FOLDER, TRAINING_HISTORY_PATH
from config.model_config import (
    MODEL_NAME, WEIGHTS_PATH, MAX_LENGTH,
    NUM_EPOCHS, TEST_SPLIT, BATCH_SIZE,
    SAVING_WEIGHTS_FREQUENCY, EARLY_STOPPING_PATIENCE,
    LR_DECAY_RATE, LEARNING_RATE, DECAYS_PER_EPOCH
)

"""
Этот код предназначен для обучения модели BERT для классификации текстов. 
Обучение выполняется на двух наборах данных: только с парсингом и с переводом аббревиатур.
Модель сохраняет веса во время обучения и проверяется на тестовых данных.

Ключевые компоненты:
1. `CustomSaver`: Кастомный обратный вызов для сохранения весов модели после каждой эпохи.
2. Подготовка данных: загрузка, предобработка и разделение на обучающую и тестовую выборки.
3. Обучение модели BERT.
4. Тестирование модели и визуализация истории обучения.
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
        # Создание папки, если она не существует
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
            # Убедиться, что метрики доступны
            if accuracy is not None and loss is not None:
                filename = f"model_{accuracy:.4f}_{loss:.4f}_{epoch+1}.h5"
                filepath = os.path.join(self.folder_name, filename)
                # Сохранение весов модели
                self.model.save_weights(filepath)
                print(f"\nВеса модели сохранены в: {filepath}")
            else:
                print("\nМетрики accuracy или loss недоступны.")


# ====== Обучение на парсированных данных ======

# Параметры

# Remove JSON loading and use MODEL_CONFIG directly
DATA_PATH = os.path.join(DATASETS_FOLDER, 'data_jan10_combined_parsed.csv')

# Загрузка данных
df = pd.read_csv(DATA_PATH, sep='|').fillna('empty')
df['clear_name'] = df['clear_name'].apply(lambda x: x.lower())
df['product_name'] = df['product_name'].apply(lambda x: x.lower())
df.loc[df['clear_name'] != 'несъедобное', 'clear_name'] = 'съедобное'
print("Длина датасета:", len(df))

# Подготовка данных
x = df['product_name'].tolist()
y = df['clear_name'].tolist()

le = LabelEncoder()
le.fit(y)
y = le.transform(y)
NUM_CLASSES = len(le.classes_)
print("Количество классов:", NUM_CLASSES)

# Разделение данных на обучающую и тестовую выборки
x_train, x_test, y_train, y_test = train_test_split(
    x, 
    y, 
    test_size=TEST_SPLIT, 
    random_state=42, 
    stratify=y
)

# Инициализация модели
bert_model = BertModel(
    num_labels=NUM_CLASSES,
    max_length=MAX_LENGTH
)

# Настройка путей сохранения
model_full_name = f"bert_binary_weights_{MODEL_NAME}_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M')}"
saving_path = os.path.join(TRAINING_HISTORY_PATH, model_full_name)

# Настройка обратных вызовов
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE)
callbacks = [
    CustomSaver(save_freq=SAVING_WEIGHTS_FREQUENCY, folder_name=saving_path),
    es
]

print('decay_steps used:',(len(x_train)//BATCH_SIZE) // DECAYS_PER_EPOCH)

# Обучение модели
history = bert_model.train(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    batch_size=BATCH_SIZE,
    lr = LEARNING_RATE,
    epochs=NUM_EPOCHS,
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')],
    callbacks=callbacks,
    lr_decay=True,
    decay_rate = LR_DECAY_RATE, # Reduce lr by 4% every time
    decay_steps= (len(x_train)//BATCH_SIZE) // DECAYS_PER_EPOCH # Reduce lr every half an epoch
)
# Тестирование модели
print('Тестирование модели')
history_train, history_test = model_evaluate_during_training(
    bert_model,
    saving_path,
    train_data=(x_train, y_train),
    test_data=(x_test, y_test),
    save_history=True
)

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
destination_path = os.path.join(WEIGHTS_PATH, model_full_name + '.h5')
os.makedirs(WEIGHTS_PATH, exist_ok=True)  # Ensure directory exists
shutil.copy(best_weights_path, destination_path)
