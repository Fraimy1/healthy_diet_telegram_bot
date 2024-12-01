from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import numpy as np

"""
Этот код реализует использование BERT для классификации текстовых данных. 
Дополнительно включены классы для использования логистической регрессии, бинарной классификации и обработки меток.

Ключевые компоненты:
1. `BertModel`: Основной класс для работы с моделью BERT, включая обучение, оценку и предсказание.
2. `BertModelWithLogisticRegression`: Комбинация BERT и логистической регрессии для уточнения предсказаний.
3. `BertWithBinaryBert`: Использование двух моделей BERT для бинарной и многоуровневой классификации.
4. `CustomLabelEncoder`: Расширенная версия `LabelEncoder` для обработки класса "несъедобное".

Применение:
- Классификация текстовых данных с использованием модели BERT и её комбинаций с другими методами.
"""

class BertModel:
    """
    Класс для работы с моделью BERT для задачи классификации.
    """
    def __init__(self, num_labels, max_length=150, use_gpu=True, model_name='bert-base-uncased'):
        """
        Инициализация модели BERT.

        Параметры:
        - num_labels (int): Количество классов.
        - max_length (int): Максимальная длина входного текста.
        - use_gpu (bool): Использовать ли GPU для вычислений.
        - model_name (str): Название предобученной модели.
        """
        self._model_name = model_name
        self._max_length = max_length
        self._device = '/CPU:0'
        if use_gpu:
            if len(tf.config.list_physical_devices('GPU')) > 0:
                self._device = '/GPU:0'
            else:
                print("Внимание: GPU недоступен, используется CPU.")
        self.model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.history = None
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self._SEP_TOKEN = self.tokenizer.sep_token
        self.outputs_confidences = True

    def tokenize(self, texts):
        """
        Токенизация входных текстов.

        Параметры:
        - texts (list[str]): Список текстов для токенизации.

        Возвращает:
        - dict: Токенизированные данные.
        """
        return self.tokenizer(
            texts,
            add_special_tokens=True,
            max_length=self._max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='tf'
        )

    def _build_dataset(self, inputs, outputs, batch_size):
        """
        Создание датасета для обучения.

        Параметры:
        - inputs (list): Входные данные.
        - outputs (list): Метки классов.
        - batch_size (int): Размер мини-пакета.

        Возвращает:
        - tf.data.Dataset: Подготовленный датасет.
        """
        tokenized_inputs = self.tokenize(inputs)
        labels = tf.convert_to_tensor(outputs, dtype=tf.int32)
        dataset = tf.data.Dataset.from_tensor_slices((dict(tokenized_inputs), labels))
        dataset = dataset.shuffle(len(outputs)).batch(batch_size)
        return dataset

    def train(self, inputs, outputs, epochs, validation_data=None, lr=1e-4, batch_size=16, 
              metrics=['accuracy'], callbacks=None):
        """
        Обучение модели BERT.

        Параметры:
        - inputs (list[str]): Входные данные.
        - outputs (list[int]): Метки классов.
        - epochs (int): Количество эпох.
        - validation_data (tuple): Валидационные данные.
        - lr (float): Скорость обучения.
        - batch_size (int): Размер мини-пакета.
        - metrics (list): Метрики для оценки.
        - callbacks (list): Список обратных вызовов.

        Возвращает:
        - history: История обучения.
        """
        train_data = self._build_dataset(inputs, outputs, batch_size)

        if validation_data:
            validation_data = self._build_dataset(validation_data[0], validation_data[1], batch_size)

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        history = self.model.fit(train_data, epochs=epochs, callbacks=callbacks, validation_data=validation_data)
        self.history = history
        return history

    def evaluate(self, inputs, outputs, metrics=['accuracy'], batch_size=1):
        """
        Оценка модели.

        Параметры:
        - inputs (list[str]): Тестовые данные.
        - outputs (list[int]): Метки классов.
        - metrics (list): Метрики для оценки.
        - batch_size (int): Размер мини-пакета.

        Возвращает:
        - результаты оценки.
        """
        test_data = self._build_dataset(inputs, outputs, batch_size)
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=metrics)
        return self.model.evaluate(test_data)

    def save(self, path):
        """
        Сохранение модели и токенизатора.

        Параметры:
        - path (str): Путь для сохранения.
        """
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_weights(self, path):
        """
        Загрузка весов модели.

        Параметры:
        - path (str): Путь к файлу весов.
        """
        self.model.load_weights(path)

    def predict(self, inputs, batch_size=1):
        """
        Предсказание классов и доверительных значений.

        Параметры:
        - inputs (list[str]): Входные данные.
        - batch_size (int): Размер мини-пакета.

        Возвращает:
        - tuple: Предсказанные классы и их доверительные значения.
        """
        encoded_input = self.tokenize(inputs)
        dataset = tf.data.Dataset.from_tensor_slices(encoded_input).batch(batch_size)

        all_predicted_classes = []
        all_confidences = []

        for batch in dataset:
            logits = self.model(batch).logits
            probabilities = tf.nn.softmax(logits, axis=-1)
            predicted_class = tf.argmax(probabilities, axis=1).numpy()
            confidence = tf.reduce_max(probabilities, axis=1).numpy()

            all_predicted_classes.extend(predicted_class)
            all_confidences.extend(confidence)

        return all_predicted_classes, all_confidences

class BertModelWithLogisticRegression:
    """
    Класс для комбинированного использования модели BERT и логистической регрессии.
    Используется для предварительной фильтрации данных логистической регрессией и последующей обработки BERT.
    """
    def __init__(self, bert_model: BertModel, logistic_regression, vectorizer, inedible_class):
        """
        Инициализация класса.

        Параметры:
        - bert_model (BertModel): Модель BERT для классификации.
        - logistic_regression: Логистическая регрессия для предварительной классификации.
        - vectorizer: Векторизатор для подготовки текстовых данных для логистической регрессии.
        - inedible_class: Номер класса "несъедобное".
        """
        self.bert_model = bert_model
        self.logistic_regression = logistic_regression
        self.vectorizer = vectorizer
        self.inedible_class = inedible_class
        self.outputs_confidences = True

    def predict(self, x, batch_size=32):
        """
        Комбинированное предсказание с использованием логистической регрессии и BERT.

        Параметры:
        - x (list): Входные данные.
        - batch_size (int): Размер мини-пакета для BERT.

        Возвращает:
        - tuple: Предсказанные классы и доверительные значения.
        """
        x_transformed = self.vectorizer.transform(x)
        logistic_pred = self.logistic_regression.predict(x_transformed)

        bert_input_indices = np.where(logistic_pred == 1)[0]
        bert_input = [x[i] for i in bert_input_indices]

        final_predictions = np.full(len(x), self.inedible_class)
        final_confidences = np.full(len(x), 1.0)
        if bert_input:
            bert_predictions, confidences = self.bert_model.predict(bert_input, batch_size=batch_size)  

            for idx, prediction, confidence in zip(bert_input_indices, bert_predictions, confidences):
                final_predictions[idx] = prediction
                final_confidences[idx] = confidence

        return final_predictions, final_confidences

    def load_weights(self, path):
        """
        Загрузка весов модели BERT.

        Параметры:
        - path (str): Путь к файлу весов.
        """
        self.bert_model.load_weights(path)


class BertWithBinaryBert:
    """
    Класс для комбинированного использования бинарной модели BERT и многоуровневой модели BERT.
    """
    def __init__(self, bert_model, binary_bert, inedible_class):
        """
        Инициализация класса.

        Параметры:
        - bert_model: Основная модель BERT для многоуровневой классификации.
        - binary_bert: Модель BERT для бинарной классификации.
        - inedible_class: Номер класса "несъедобное".
        """
        self.bert_model = bert_model
        self.binary_bert = binary_bert
        self.inedible_class = inedible_class
        self.outputs_confidences = True

    def predict(self, x, batch_size=32):
        """
        Комбинированное предсказание с использованием двух моделей BERT.

        Параметры:
        - x (list): Входные данные.
        - batch_size (int): Размер мини-пакета.

        Возвращает:
        - tuple: Предсказанные классы и доверительные значения.
        """
        is_single_instance = False
        if not isinstance(x, (list, np.ndarray)):
            x = [x]
            is_single_instance = True

        binary_preds, binary_confidences = self.binary_bert.predict(x, batch_size=batch_size)
        binary_preds = np.array(binary_preds)
        binary_confidences = np.array(binary_confidences)

        bert_input_indices = np.where(binary_preds == 1)[0]
        bert_input = [x[i] for i in bert_input_indices] if len(bert_input_indices) > 0 else []

        final_predictions = np.full(len(x), self.inedible_class)
        final_confidences = binary_confidences.copy()

        if len(bert_input) > 0:
            bert_predictions, bert_confidences = self.bert_model.predict(bert_input, batch_size=batch_size)
            bert_predictions = np.array(bert_predictions)
            bert_confidences = np.array(bert_confidences)

            for idx, prediction, confidence in zip(bert_input_indices, bert_predictions, bert_confidences):
                final_predictions[idx] = prediction
                final_confidences[idx] *= confidence

        if is_single_instance:
            return final_predictions[0], final_confidences[0]
        else:
            return final_predictions, final_confidences

    def load_weights(self, path):
        """
        Загрузка весов модели BERT.

        Параметры:
        - path (str): Путь к файлу весов.
        """
        self.bert_model.load_weights(path)


class CustomLabelEncoder(LabelEncoder):
    """
    Расширенный LabelEncoder для обработки класса "несъедобное".
    """
    def __init__(self, inedible_class_num, inedible_class_name):
        """
        Инициализация класса.

        Параметры:
        - inedible_class_num (int): Код класса "несъедобное".
        - inedible_class_name (str): Название класса "несъедобное".
        """
        super().__init__()
        self.inedible_class_num = inedible_class_num
        self.inedible_class_name = inedible_class_name

    def transform(self, y):
        """
        Трансформирует метки классов, заменяя "несъедобное" на соответствующий код.

        Параметры:
        - y (list): Метки классов.

        Возвращает:
        - np.ndarray: Трансформированные метки классов.
        """
        y_transformed = np.array(y, dtype=object)
        inedible_mask = y_transformed == self.inedible_class_name
        y_transformed[inedible_mask] = self.inedible_class_num
        y_transformed[~inedible_mask] = super().transform(y_transformed[~inedible_mask])
        return y_transformed.astype(int)

    def inverse_transform(self, y):
        """
        Обратное преобразование меток классов, возвращая "несъедобное" для соответствующего кода.

        Параметры:
        - y (list): Трансформированные метки классов.

        Возвращает:
        - np.ndarray: Оригинальные метки классов.
        """
        y_transformed = np.array(y, dtype=object)
        non_inedible_mask = y_transformed != self.inedible_class_num
        if non_inedible_mask.any():
            y_transformed[non_inedible_mask] = super().inverse_transform(
                y_transformed[non_inedible_mask].astype(int)
            )
        y_transformed[y_transformed == self.inedible_class_num] = self.inedible_class_name
        return y_transformed
