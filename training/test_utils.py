"""
Описание файла
===============

Этот файл содержит набор инструментов для обучения, тестирования и оценки производительности моделей на основе архитектуры BERT. 
Он предоставляет функции для обработки данных, расчета метрик, визуализации результатов и сравнения моделей. 

Основное назначение:
--------------------
1. Предсказания на основе текста с использованием моделей BERT.
2. Вычисление метрик качества модели (точность, полнота, F1-метрика и др.).
3. Оценка производительности модели на эпохах и в процессе обучения.
4. Визуализация результатов обучения (графики истории обучения, матрицы ошибок и т.д.).
5. Анализ и визуализация неправильных предсказаний.
6. Сравнение нескольких моделей на основе их истории обучения.

Основные зависимости:
---------------------
- **Matplotlib**: Для построения графиков и визуализации.
- **Scikit-learn**: Для расчета метрик качества и построения матрицы ошибок.
- **Seaborn**: Для создания улучшенных визуализаций данных.
- **Pandas и NumPy**: Для работы с данными.
- **Tabulate**: Для удобного форматирования данных в таблицах.
- **BertModel**: Кастомная реализация модели BERT для работы с текстами.

Описание функций:
-----------------
- `predict`: Делает предсказание на основе входного текста.
- `calculate_metrics`: Вычисляет метрики качества модели.
- `test_model_on_epoch`: Тестирует модель на указанной эпохе.
- `model_evaluate_during_training`: Проводит оценку модели в процессе обучения.
- `plot_training_history`: Строит графики истории обучения.
- `plot_confusion_matrix`: Создает матрицу ошибок с аннотациями.
- `find_best_epoch`: Определяет лучшую эпоху на основе заданной метрики.
- `test_model_on_set`: Оценивает модель на заданных данных и выводит результаты.
- `plot_incorrect_predictions`: Визуализирует распределение правильных и неправильных предсказаний.
- `compare_models`: Сравнивает несколько моделей на основе их истории обучения.

Использование:
--------------
Этот файл предназначен для использования в проектах по обработке естественного языка (NLP), где требуется разработка и тестирование моделей классификации текста. 
Все функции могут быть адаптированы под конкретные задачи, такие как бинарная классификация или многоклассовая классификация.
"""
import matplotlib.pyplot as plt
import time
import os
import json
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
from tabulate import tabulate
import pandas as pd
from collections import defaultdict
from model.bert_model import BertModel
import itertools
from config.model_config import BATCH_SIZE

def predict(text, label_encoder, model: BertModel):
    """
    Делает предсказание на основе переданного текста, используя заданную модель и кодировщик меток.

    Параметры
    ---------
    text : str
        Текст, для которого нужно сделать предсказание.
    label_encoder : sklearn.preprocessing.LabelEncoder
        Кодировщик меток для декодирования предсказанной метки.
    model : BertModel
        Модель, используемая для предсказания.

    Возвращает
    ----------
    None
    """
    pred_text, pred_probability = model.predict([text])
    pred_text = [pred_text] if type(pred_text) != list else pred_text
    pred_text = label_encoder.inverse_transform(pred_text)
    print('Предсказание:', pred_text[0], '|', 'Уверенность:', pred_probability[0])


def calculate_metrics(y_true, y_pred):
    """
    Вычисляет метрики на основе истинных и предсказанных меток.

    Параметры
    ---------
    y_true : array_like
        Истинные метки.
    y_pred : array_like
        Предсказанные метки.

    Возвращает
    ----------
    accuracy : float
        Точность модели (accuracy).
    precision : float
        Точность (precision) модели, вычисленная как взвешенное среднее точностей каждого класса.
    recall : float
        Полнота (recall) модели, вычисленная как взвешенное среднее полнот каждого класса.
    f1 : float
        F1-мера (F1 score) модели, вычисленная как взвешенное среднее F1-мер каждого класса.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    return accuracy, precision, recall, f1

def test_model_on_epoch(x, y, model: BertModel, path, batch_size=BATCH_SIZE):
    """
    Оценивает заданную модель на указанных данных для указанной эпохи.

    Параметры
    ---------
    x : array_like
        Входные данные для оценки модели.
    y : array_like
        Истинные метки для входных данных.
    model : BertModel
        Модель для оценки.
    path : str
        Путь к весам модели, которые нужно загрузить.
    batch_size : int
        Размер батча, используемый для оценки.

    Возвращает
    ----------
    accuracy : float
        Точность модели (accuracy) на указанных данных для текущей эпохи.
    precision : float
        Точность (precision) модели на указанных данных для текущей эпохи.
    recall : float
        Полнота (recall) модели на указанных данных для текущей эпохи.
    f1 : float
        F1-мера (F1 score) модели на указанных данных для текущей эпохи.
    """
    model.load_weights(path)
    preds, _ = model.predict(x, batch_size=batch_size)
    accuracy, precision, recall, f1 = calculate_metrics(y, preds) 
    return accuracy, precision, recall, f1

def model_evaluate_during_training(model: BertModel, training_path, epochs=None, train_data=None, test_data=None, batch_size=BATCH_SIZE, save_history=True):
    """
    Оценивает заданную модель на предоставленных данных на каждой эпохе в процессе обучения.

    Параметры
    ---------
    model : BertModel
        Модель для оценки.
    training_path : str
        Путь к весам модели, сохранённым в процессе обучения.
    epochs : int
        Количество эпох для оценки модели.
    train_data : tuple
        Кортеж с тренировочными данными.
    test_data : tuple
        Кортеж с тестовыми данными.
    save_history : bool
        Сохранять ли историю обучения в файл.

    Возвращает
    ----------
    history_train : dict
        История обучения модели на тренировочных данных.
    history_test : dict
        История обучения модели на тестовых данных.
    """
    assert train_data is not None or test_data is not None, "At least one of train_data or test_data must be provided."
    if epochs is None:
        epochs = len(os.listdir(training_path))
    
    sorting_key = lambda x: int(x.split('_')[-1].split('.')[0])
    history_train = {'accuracy': [], 'precision': [], 'recall': [], 'f1-score': []}
    history_test = {'accuracy': [], 'precision': [], 'recall': [], 'f1-score': []}
    metrics_names = ['accuracy', 'precision', 'recall', 'f1-score']

    for path in sorted(os.listdir(training_path), key=sorting_key)[:epochs]:
        epoch = sorting_key(path)
        print(f"\033[92mEpoch {epoch}/{epochs}\033[0m")
        start_time = time.time()

        # Расчёт метрик для тренировочных данных
        if train_data:
            train_metrics = test_model_on_epoch(train_data[0], train_data[1], model, os.path.join(training_path, path), batch_size=batch_size) 
            for metric, value in zip(metrics_names, train_metrics):
                history_train[metric].append(value)
        
        # Расчёт метрик для тестовых данных
        if test_data:
            test_metrics = test_model_on_epoch(test_data[0], test_data[1], model, os.path.join(training_path, path), batch_size=batch_size)
            for metric, value in zip(metrics_names, test_metrics):
                history_test[metric].append(value)

        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"\033[36mEpoch {epoch} completed in {elapsed_time:.2f} seconds\033[0m")
        print(f"\033[37mTrain Accuracy: {train_metrics[0]:.4f}, Precision: {train_metrics[1]:.4f}, Recall: {train_metrics[2]:.4f}, F1-score: {train_metrics[3]:.4f}\033[0m") if train_data else None
        print(f"\033[37mTest Accuracy: {test_metrics[0]:.4f}, Precision: {test_metrics[1]:.4f}, Recall: {test_metrics[2]:.4f}, F1-score: {test_metrics[3]:.4f}\033[0m") if test_data else None
    
    if save_history:
        history = {
            'train': history_train,
            'test': history_test
        }
        history_file = os.path.join(training_path, f'training_history_{epochs}_epochs.json')
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=4)
        print(f"Training history saved to {history_file}")

    return history_train, history_test

def plot_training_history(history_train=None, history_test=None, path_to_json = None, saving_path:str=None, show_plot=False):
    """
    Построить график истории обучения модели.

    Параметры
    ---------
    history_train : dict
        История обучения модели.
    history_test : dict
        История тестирования модели.
    path_to_json : str
        Путь к файлу json, содержащему историю обучения.
    saving_path : str
        Путь для сохранения графика. Если None, график не будет сохранён.
    show_plot : bool
        Показывать ли график. Если False, график не будет показан.

    Возвращает
    ---------
    None
    """
    if path_to_json:
        if history_train is not None or history_test is not None:
            print('Warning: Both path_to_json and history_train/history_test are provided. Using path_to_json.')
        
        with open(path_to_json, 'r') as f:
            history = json.load(f)
        history_train = history.get('train', None)
        history_test = history.get('test', None)

    assert history_train is not None or history_test is not None, "At least one of history_train or history_test must be provided."
    metrics = ['accuracy', 'precision', 'recall', 'f1-score']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        if history_train and metric in history_train:
            ax.plot(history_train[metric], label=f'Training {metric}', color='blue')
        if history_test and metric in history_test:
            ax.plot(history_test[metric], label=f'Test {metric}', color='orange')
        
        ax.set_xlabel('Epochs')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'Training and Test {metric.capitalize()} Over Epochs')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    if saving_path:
        plt.savefig(saving_path)
    if show_plot:
        plt.show()


def plot_confusion_matrix(model:BertModel, weights_path, num_classes, train_data=None, 
                          test_data=None, batch_size=64, show_plot=True, saving_path=None,
                          label_encoder=None):
    """
    Построить матрицу ошибок модели для обучающих или тестовых данных, 
    или для обоих. Матрица ошибок нормализована по строкам для отображения цветов, 
    а фактические значения аннотированы на графике.
    Предупреждение:
    Не использовать с моделями с большим количеством классов (больше 100).
    Параметры
    ----------
    model : keras.Model
        Модель для оценки.
    weights_path : str
        Путь к файлу с весами модели.
    num_classes : int
        Количество классов в задаче.
    train_data : tuple, optional
        Обучающие данные. Если не предоставлены, функция построит только для тестовых данных.
    test_data : tuple, optional
        Тестовые данные. Если не предоставлены, функция построит только для обучающих данных.
    batch_size : int, optional
        Размер пакета для предсказаний. По умолчанию 64.
    show_plot : bool, optional
        Показывать ли график. По умолчанию True.
    saving_path : str, optional
        Путь для сохранения графика. Если не предоставлен, график не будет сохранён.
    label_encoder : sklearn.preprocessing.LabelEncoder, optional
        Кодировщик меток для получения имен классов. Если не предоставлен, будут использоваться числовые индексы классов.
    """
    assert train_data is not None or test_data is not None, "At least one of train_data or test_data must be provided."
    
    model.load_weights(weights_path)
    
    # Динамически регулируем размер графика в зависимости от числа классов
    fig_size = max(10, num_classes // 10)  # Обеспечиваем минимальный размер 10, масштабируем в зависимости от num_classes

    # Получаем имена классов, если передан label_encoder, иначе используем числовые индексы классов
    if label_encoder:
        class_names = label_encoder.inverse_transform(np.arange(num_classes))
        # Сортируем имена классов по их числовому эквиваленту
        sorted_indices = np.argsort(label_encoder.transform(class_names))
        class_names = class_names[sorted_indices]
    else:
        class_names = np.arange(num_classes)

    if train_data:
        x_train, y_train = train_data[0], train_data[1]
        pred_train, _ = model.predict(x_train, batch_size=batch_size)
        print('Predictions were made, creating cm_matrix')
        cm_train = confusion_matrix(y_train, pred_train)
        
        # Нормализуем матрицу ошибок по строкам для цветов
        cm_train_normalized = (cm_train.astype('float') / max(cm_train.sum(axis=1)[:, np.newaxis], 1))
        print('Plotting cm matrix')
        plt.figure(figsize=(fig_size, fig_size))
        sns.heatmap(cm_train_normalized, annot=cm_train, fmt="d", cmap="Blues", 
                    xticklabels=class_names, yticklabels=class_names)
        plt.xticks(rotation=90)  # Поворот меток по оси X, чтобы избежать наложения
        plt.title("Confusion Matrix - Train Data (Colors Normalized, Values Actual)")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        if saving_path:
            plt.savefig(saving_path)
        if show_plot:
            plt.show()

    if test_data:
        x_test, y_test = test_data[0], test_data[1]
        pred_test, _ = model.predict(x_test, batch_size=batch_size)
        cm_test = confusion_matrix(y_test, pred_test)
        print('Predictions were made, creating cm_matrix')
        # Нормализуем матрицу ошибок по строкам для цветов
        cm_test_normalized = (cm_test.astype('float') / max(cm_test.sum(axis=1)[:, np.newaxis]))
        print('Plotting cm matrix')
        plt.figure(figsize=(fig_size, fig_size))
        sns.heatmap(cm_test_normalized, annot=cm_test, fmt="d", cmap="Blues", 
                    xticklabels=class_names, yticklabels=class_names)
        plt.xticks(rotation=90)  # Поворот меток по оси X, чтобы избежать наложения
        plt.title("Confusion Matrix - Test Data (Colors Normalized, Values Actual)")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        if saving_path:
            plt.savefig(saving_path)
        if show_plot:
            plt.show()



def find_best_epoch(path_to_json=None, history_train=None, history_test=None,

                     deciding_metric='f1-score', deciding_set='test', verbose=True):
    """
    Найти лучшую эпоху на основе заданной метрики и набора данных.

    Параметры
    ----------
    path_to_json : str
        Путь к JSON файлу, содержащему историю обучения.
    history_train : dict
        История обучения модели.
    history_test : dict
        История тестирования модели.
    deciding_metric : str
        Метрика для определения лучшей эпохи. Выберите из 'accuracy', 'precision', 'recall', 'f1-score'.
    deciding_set : str
        Набор данных, на основе которого будет определена лучшая эпоха. Выберите из 'train' или 'test'.
    verbose : bool
        Печать ли лучшей эпохи и соответствующих метрик.

    Возвращаемое значение
    -------
    best_epoch : int
        Лучшая эпоха на основе заданной метрики и набора данных.
    """
    assert deciding_set in ['train', 'test'], "deciding_set должен быть либо 'train', либо 'test'."
    assert deciding_metric in ['accuracy', 'precision', 'recall', 'f1-score'], \
        "Неверная deciding_metric. Выберите из 'accuracy', 'precision', 'recall', 'f1-score'."

    if path_to_json is None:
        history = {'train': history_train, 'test': history_test}
        assert history_train is not None or history_test is not None, "Необходимо предоставить хотя бы одну из историй: history_train или history_test."
    else:
        if history_train is not None or history_test is not None:
            print('Warning: Both path_to_json and history_train/history_test are provided. Using path_to_json.')

        with open(path_to_json, 'r') as f:
            history = json.load(f)
        history_train = history.get('train', None)
        history_test = history.get('test', None)
        assert history_train is not None or history_test is not None, "No training or testing history found in the provided JSON file."

    assert history.get(deciding_set, None) is not None, f"No {deciding_set} history found in the provided JSON file."
    assert deciding_metric in history[deciding_set], f"Invalid deciding_metric '{deciding_metric}'. Available metrics: {list(history[deciding_set].keys())}"
    
    best_epoch = np.argmax(history[deciding_set][deciding_metric])  
    if not verbose:
        return best_epoch
    
    available_metrics = sorted(list(set(history[deciding_set].keys())))

    train_metrics = []
    test_metrics = []
    diff_metrics = [] 

    if history_train:
        train_metrics = [(metric, history_train[metric][best_epoch]) for metric in available_metrics]

    if history_test:
        test_metrics = [(metric, history_test[metric][best_epoch]) for metric in available_metrics]
        
    if history_train and history_test:
        diff_metrics = [(metric, history_test[metric][best_epoch] - history_train[metric][best_epoch]) for metric in available_metrics]

    num_epochs = len(history[deciding_set][deciding_metric])
    print(f"\nBest Epoch: {best_epoch + 1} (out of {num_epochs})\n")
    if train_metrics:
        print("Train Metrics on Best Epoch:")
        print(tabulate(train_metrics, headers=["Metric", "Value"], tablefmt="fancy_grid", floatfmt=".4f"))
        print("\n")

    if test_metrics:
        print("Test Metrics on Best Epoch:")
        print(tabulate(test_metrics, headers=["Metric", "Value"], tablefmt="fancy_grid", floatfmt=".4f"))
        print("\n")

    if diff_metrics:
        print("Difference (Test - Train) on Best Epoch:")
        print(tabulate(diff_metrics, headers=["Metric", "Difference"], tablefmt="fancy_grid", floatfmt="+.4f"))

    return best_epoch

def test_model_on_set(x, y, model:BertModel, weights_path=None, batch_size=256, label_encoder=None,
                       plain_text=False, show:str='both', saving_path:str=None):
    """
    Оценить модель на заданных данных и вывести результаты в табличном формате.

    Параметры
    ----------
    x : array-like
        Входные данные для оценки.
    y : array-like
        Истинные метки входных данных.
    model : keras.Model
        Модель для оценки.
    weights_path : str, optional
        Путь к весам модели для загрузки. Если не указан, используются текущие веса модели.
    batch_size : int, optional
        Размер пакета для предсказания. По умолчанию 256.
    label_encoder : sklearn.preprocessing.LabelEncoder, optional
        Кодировщик меток для получения имен классов. Если не указан, будут использоваться индексы классов.
    plain_text : bool, optional
        Нужно ли выводить таблицу в формате обычного текста. По умолчанию False.
    show : str, optional
        Какие предсказания показывать. Может быть 'correct' (правильные), 'incorrect' (неправильные) или 'both' (все). По умолчанию 'both'.
    saving_path : str, optional
        Путь для сохранения таблицы в виде Excel файла. Если не указан, таблица не будет сохранена.

    Возвращает
    -------
    None
    """
    assert show in ['correct', 'incorrect', 'both'], "show должен быть 'correct', 'incorrect' или 'both'."
    
    if weights_path:
        model.load_weights(weights_path)
    
    if hasattr(model, 'outputs_confidences') and model.outputs_confidences:
        preds, confidences = model.predict(x, batch_size=batch_size)
    else:
        preds = model.predict(x, batch_size=batch_size)
        confidences = None  # Нет доступных уверенности
    
    if label_encoder:
        preds_text = label_encoder.inverse_transform(preds)
        y_text = label_encoder.inverse_transform(y)

    table = []
    for i in range(len(preds)):
        confidence = round(confidences[i],4) if confidences is not None else 'N/A'  # По умолчанию 'N/A', если уверенность отсутствует
        prediction = preds[i]
        true_value = y[i]
        row = None
        if (prediction == true_value) and show in ['correct', 'both']:
            if plain_text:
                row = [f"{x[i][:80]}", f"{preds_text[i]}", f"{confidence}", f"{y_text[i]}", 'correct']  
            else:
                row = [f"\033[32m{x[i][:80]}\033[0m",f"\033[32m{preds_text[i]}\033[0m", f"\033[32m{confidence}\033[0m", f"\033[32m{y_text[i]}\033[0m"]  # Зеленый для правильных

        if (prediction != true_value) and show in ['incorrect', 'both']:
            if plain_text:
                row = [f"{x[i][:80]}", f"{preds_text[i]}", f"{confidence}", f"{y_text[i]}", 'incorrect']
            else:
                row = [f"\033[31m{x[i][:80]}\033[0m",f"\033[31m{preds_text[i]}\033[0m", f"\033[31m{confidence}\033[0m", f"\033[31m{y_text[i]}\033[0m"]  # Красный для неправильных
        
        if row:
            table.append(row)

    if saving_path:
        df = pd.DataFrame({'Input': x, 'Prediction': preds_text, 'Confidence': confidences,
                            'True Value': y_text, 'Correctness': ['Correct' if preds[i] == y[i] else 'Incorrect' for i in range(len(preds))]})
        df.to_excel(saving_path, index=False)

    # Печать табличного результата
    if plain_text:
        print(tabulate(table, headers=["Input", "Prediction", "Confidence", "True Value", 'Correctness'], tablefmt="fancy_grid"))
    else:
        print(tabulate(table, headers=["Input", "Prediction", "Confidence", "True Value"], tablefmt="fancy_grid"))


def plot_incorrect_predictions(x, y, model:BertModel, weights_path=None, batch_size=256, label_encoder=None, saving_path=None, show_plot=True):
    """
    Построить график количества правильных и неправильных предсказаний для каждого класса в виде гистограммы,
    с логарифмической шкалой для оси X.
    
    Параметры
    ----------
    x : array-like
        Входные данные для предсказания.
    y : array-like
        Истинные метки данных.
    model : keras.Model
        Модель для оценки.
    weights_path : str, optional
        Путь к весам модели для загрузки.
    batch_size : int, optional
        Размер пакета для предсказания. По умолчанию 256.
    label_encoder : sklearn.preprocessing.LabelEncoder, optional
        Кодировщик меток для получения имен классов. Если не указан, будут использоваться индексы классов.
    saving_path : str, optional
        Путь для сохранения графика. Если не указан, график не будет сохранен.
    show_plot : bool, optional
        Нужно ли показывать график. По умолчанию True.
    """
    if weights_path:
        model.load_weights(weights_path)

    if hasattr(model, 'outputs_confidences') and model.outputs_confidences:
        preds, confidences = model.predict(x, batch_size=batch_size)
    else:
        preds = model.predict(x, batch_size=batch_size)
        confidences = None  # Нет доступных уверенности

    if label_encoder:
        y_text = label_encoder.inverse_transform(y)
        preds_text = label_encoder.inverse_transform(preds)
    else:
        y_text = y
        preds_text = preds

    class_counts = defaultdict(lambda: {'correct': 0, 'incorrect': 0})

    for i in range(len(preds_text)):
        true_label = y_text[i]
        predicted_label = preds_text[i]

        if predicted_label == true_label:
            class_counts[true_label]['correct'] += 1
        else:
            class_counts[true_label]['incorrect'] += 1

    # Подготовка данных для построения графика
    data = []
    for class_name, counts in class_counts.items():
        data.append([class_name, counts['correct'], counts['incorrect'], counts['correct'] + counts['incorrect']])

    # Создание DataFrame для удобства построения
    df = pd.DataFrame(data, columns=['Class', 'Correct Count', 'Incorrect Count', 'Total Count'])

    # Сортировка данных по общему количеству для улучшения читаемости
    df_sorted = df.sort_values('Total Count', ascending=True)

    # Установка размера фигуры
    plt.figure(figsize=(20, 140))

    # Определение позиций и ширины баров
    bar_width = 0.7  # Уменьшенная ширина баров
    spacing = 0.15  # Увеличенное расстояние между барами
    y_positions = np.arange(len(df_sorted))

    # Построение баров для правильных (зеленых) и неправильных (красных) предсказаний с логарифмической шкалой
    plt.barh(y_positions, df_sorted['Correct Count'], color='green', edgecolor='black', height=bar_width, label='Correct')
    plt.barh(y_positions, df_sorted['Incorrect Count'], left=df_sorted['Correct Count'], color='red', edgecolor='black', height=bar_width, label='Incorrect')

    # Логарифмическая шкала для оси X
    plt.xscale('log')

    # Настройка меток по оси Y с увеличенным расстоянием
    plt.gca().set_yticks(y_positions + (spacing / 2))  # Добавление расстояния к позициям
    plt.gca().set_yticklabels(df_sorted['Class'])

    # Эстетика и подписи
    plt.xlabel('Количество предсказаний (логарифмическая шкала)')
    plt.title('Правильные и неправильные предсказания по классам')
    plt.legend(title='Prediction')

    # Сохранение или отображение графика
    plt.tight_layout()
    if saving_path:
        plt.savefig(saving_path)
    if show_plot:
        plt.show()


def compare_models(training_paths: list[str], model_names:list[str]=[], show='both', show_best_result=False, saving_path=None, show_plot=True):
    """
    Сравнить производительность нескольких моделей, построив графики их истории обучения и тестирования.

    Параметры
    ----------
    training_paths : list[str]
        Пути к каталогам, содержащим истории обучения моделей для сравнения.
    model_names : list[str], optional
        Имена моделей для сравнения. Если не указаны, будут использованы имена по умолчанию "Model 1", "Model 2" и т.д.
    show : str, optional
        Какие истории отображать. Должно быть одним из 'train', 'test' или 'both'. По умолчанию 'both'.
    show_best_result : bool, optional
        Нужно ли показывать лучший результат для каждой модели. По умолчанию False.
    saving_path : str, optional
        Путь для сохранения графика. Если не указан, график не будет сохранен.
    show_plot : bool, optional
        Нужно ли отображать график. По умолчанию True.
    """
    if not model_names:
        model_names = [f"Model {i+1}" for i in range(len(training_paths))]
    else:
        assert len(model_names) == len(training_paths), "Количество имен моделей должно совпадать с количеством путей обучения."

    assert show in ['train', 'test', 'both'], "Параметр 'show' должен быть 'train', 'test' или 'both'."

    # Инициализация словаря для хранения историй для каждой модели
    models_histories = {}

    # Чтение файлов истории JSON для каждой модели
    for path, model_name in zip(training_paths, model_names):
        # Поиск файла JSON в пути
        json_files = [file for file in os.listdir(path) if file.endswith('.json')]
        assert json_files, f"Не найден файл истории JSON в {path}"
        json_file = json_files[0]  # Предполагаем, что один файл JSON на путь

        json_path = os.path.join(path, json_file)
        with open(json_path, 'r') as f:
            history = json.load(f)
        models_histories[model_name] = history

    # Определение метрик для построения графиков
    metrics = ['accuracy', 'precision', 'recall', 'f1-score']

    # Подготовка подграфиков
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()

    # Настройка цикла цветов
    color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    model_colors = {model_name: next(color_cycle) for model_name in model_names}

    # Построение графиков для каждой метрики
    for i, metric in enumerate(metrics):
        ax = axes[i]
        for model_name, history in models_histories.items():
            color = model_colors[model_name]
            # Проверка, есть ли данные для построения
            has_train = show in ['train', 'both'] and 'train' in history and metric in history['train']
            has_test = show in ['test', 'both'] and 'test' in history and metric in history['test']

            # Построение графика для обучения
            if has_train:
                train_metric = history['train'][metric]
                epochs = range(1, len(train_metric) + 1)
                ax.plot(
                    epochs,
                    train_metric,
                    label=f'{model_name} Train',
                    linestyle='-',
                    marker='',
                    color=color
                )
                if show_best_result:
                    # Находим максимальное значение
                    best_value = max(train_metric)
                    # Строим горизонтальную линию на уровне best_value
                    ax.axhline(y=best_value, color=color, linestyle=':', label=f'{model_name} Best Train')
            # Построение графика для тестирования
            if has_test:
                test_metric = history['test'][metric]
                epochs = range(1, len(test_metric) + 1)
                ax.plot(
                    epochs,
                    test_metric,
                    label=f'{model_name} Test',
                    linestyle='--',
                    marker='',
                    color=color
                )
                if show_best_result:
                    # Находим максимальное значение
                    best_value = max(test_metric)
                    # Строим горизонтальную линию на уровне best_value
                    ax.axhline(y=best_value, color=color, linestyle='-.', label=f'{model_name} Best Test')

            # Обработка случая, когда нет данных для обучения или тестирования
            if not has_train and not has_test:
                print(f"Предупреждение: Нет данных для отображения для метрики '{metric}' для модели '{model_name}'.")

        ax.set_xlabel('Эпохи')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'Сравнение моделей - {metric.capitalize()}')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    if saving_path:
        plt.savefig(saving_path)
    if show_plot:
        plt.show()
