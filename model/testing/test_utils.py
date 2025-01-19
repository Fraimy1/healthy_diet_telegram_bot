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
import itertools

def predict(text, label_encoder, model):
    pred_text, pred_probability = model.predict([text])
    pred_text = [pred_text] if type(pred_text) != list else pred_text
    pred_text = label_encoder.inverse_transform(pred_text)
    print('Prediction:', pred_text[0],'|', 'Confidence:', pred_probability[0])

def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    return accuracy, precision, recall, f1

def test_model_on_epoch(x, y, model, path, batch_size=256):
    model.load_weights(path)
    preds, _ = model.predict(x, batch_size=batch_size)
    accuracy, precision, recall, f1 = calculate_metrics(y, preds) 
    return accuracy, precision, recall, f1

def model_evaluate_during_training(model, training_path, epochs=None, train_data=None, test_data=None, save_history=True):
    assert train_data is not None or test_data is not None, "At least one of train_data or test_data must be provided."
    if epochs is None:
        epochs = len(os.listdir(training_path))
    
    sorting_key = lambda x: int(x.split('_')[-1].split('.')[0])
    history_train = {'accuracy':[],'precision':[],'recall':[],'f1-score':[]}
    history_test =  {'accuracy':[],'precision':[],'recall':[],'f1-score':[]}
    metrics_names = ['accuracy', 'precision', 'recall', 'f1-score']

    for path in sorted(os.listdir(training_path), key=sorting_key)[:epochs]:
        epoch = sorting_key(path)
        print(f"\033[92mEpoch {epoch}/{epochs}\033[0m")
        start_time = time.time()

        # Calculate train metrics
        if train_data:
            train_metrics = test_model_on_epoch(train_data[0], train_data[1], model, os.path.join(training_path, path)) 
            for metric, value in zip(metrics_names, train_metrics):
                history_train[metric].append(value)
        # Calculate test metrics
        if test_data:
            test_metrics = test_model_on_epoch(test_data[0], test_data[1], model, os.path.join(training_path, path))
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


def plot_confusion_matrix(model, weights_path, num_classes, train_data=None, 
                          test_data=None, batch_size=64, show_plot=True, saving_path=None,
                          label_encoder=None):
    assert train_data is not None or test_data is not None, "At least one of train_data or test_data must be provided."
    
    model.load_weights(weights_path)
    
    # Dynamically adjust figure size based on the number of classes
    fig_size = max(10, num_classes // 10)  # Ensure a minimum size of 10, scale with num_classes

    # Get class names if label_encoder is provided, else use numerical class indices
    if label_encoder:
        class_names = label_encoder.inverse_transform(np.arange(num_classes))
        # Sort class names by their numerical equivalent
        sorted_indices = np.argsort(label_encoder.transform(class_names))
        class_names = class_names[sorted_indices]
    else:
        class_names = np.arange(num_classes)

    if train_data:
        x_train, y_train = train_data[0], train_data[1]
        pred_train, _ = model.predict(x_train, batch_size=batch_size)
        print('Predictions were made, creating cm_matrix')
        cm_train = confusion_matrix(y_train, pred_train)
        
        # Normalize the confusion matrix row-wise for colors
        cm_train_normalized = (cm_train.astype('float') / max(cm_train.sum(axis=1)[:, np.newaxis], 1))
        print('Plotting cm matrix')
        plt.figure(figsize=(fig_size, fig_size))
        sns.heatmap(cm_train_normalized, annot=cm_train, fmt="d", cmap="Blues", 
                    xticklabels=class_names, yticklabels=class_names)
        plt.xticks(rotation=90)  # Rotate predicted labels to avoid overlap
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
        # Normalize the confusion matrix row-wise for colors
        cm_test_normalized = (cm_test.astype('float') / max(cm_test.sum(axis=1)[:, np.newaxis]))
        print('Plotting cm matrix')
        plt.figure(figsize=(fig_size, fig_size))
        sns.heatmap(cm_test_normalized, annot=cm_test, fmt="d", cmap="Blues", 
                    xticklabels=class_names, yticklabels=class_names)
        plt.xticks(rotation=90)  # Rotate predicted labels to avoid overlap
        plt.title("Confusion Matrix - Test Data (Colors Normalized, Values Actual)")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        if saving_path:
            plt.savefig(saving_path)
        if show_plot:
            plt.show()

#TODO: maybe make ROC curve too

def find_best_epoch(path_to_json=None, history_train=None, history_test=None,
                     deciding_metric='f1-score', deciding_set='test', verbose=True):
    assert deciding_set in ['train', 'test'], "deciding_set must be either 'train' or 'test'."
    assert deciding_metric in ['accuracy', 'precision', 'recall', 'f1-score'], \
        "Invalid deciding_metric. Choose from 'accuracy', 'precision', 'recall', 'f1-score'."

    if path_to_json is None:
        history = {'train': history_train, 'test': history_test}
        assert history_train is not None or history_test is not None, "At least one of history_train or history_test must be provided."
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

def test_model_on_set(x, y, model, weights_path=None, batch_size=256, label_encoder=None,
                       plain_text=False, show:str='both', saving_path:str=None):
    assert show in ['correct', 'incorrect', 'both'], "show must be either 'correct', 'incorrect', or 'both'."
    
    if weights_path:
        model.load_weights(weights_path)
    
    if hasattr(model, 'outputs_confidences') and model.outputs_confidences:
        preds, confidences = model.predict(x, batch_size=batch_size)
    else:
        preds = model.predict(x, batch_size=batch_size)
        confidences = None  # No confidences available
    
    if label_encoder:
        preds_text = label_encoder.inverse_transform(preds)
        y_text = label_encoder.inverse_transform(y)

    table = []
    for i in range(len(preds)):
        confidence = round(confidences[i],4) if confidences is not None else 'N/A'  # Default to 'N/A' when confidences are missing
        prediction = preds[i]
        true_value = y[i]
        row = None
        if (prediction == true_value) and show in ['correct', 'both']:
            if plain_text:
                row = [f"{x[i][:80]}", f"{preds_text[i]}", f"{confidence}", f"{y_text[i]}", 'correct']  
            else:
                row = [f"\033[32m{x[i][:80]}\033[0m",f"\033[32m{preds_text[i]}\033[0m", f"\033[32m{confidence}\033[0m", f"\033[32m{y_text[i]}\033[0m"]  # Green for correct

        if (prediction != true_value) and show in ['incorrect', 'both']:
            if plain_text:
                row = [f"{x[i][:80]}", f"{preds_text[i]}", f"{confidence}", f"{y_text[i]}", 'incorrect']
            else:
                row = [f"\033[31m{x[i][:80]}\033[0m",f"\033[31m{preds_text[i]}\033[0m", f"\033[31m{confidence}\033[0m", f"\033[31m{y_text[i]}\033[0m"]  # Red for incorrect
        
        if row:
            table.append(row)

    if saving_path:
        df = pd.DataFrame({'Input': x, 'Prediction': preds_text, 'Confidence': confidences,
                            'True Value': y_text, 'Correctness': ['Correct' if preds[i] == y[i] else 'Incorrect' for i in range(len(preds))]})
        df.to_excel(saving_path, index=False)

    # Print the tabulated result
    if plain_text:
        print(tabulate(table, headers=["Input", "Prediction", "Confidence", "True Value", 'Correctness'], tablefmt="fancy_grid"))
    else:
        print(tabulate(table, headers=["Input", "Prediction", "Confidence", "True Value"], tablefmt="fancy_grid"))

#TODO: Make a new metic that will give a % of incorrect predictions for each class in a vertical plot

def plot_incorrect_predictions(x, y, model, weights_path=None, batch_size=256, label_encoder=None, saving_path=None, show_plot=True):
    if weights_path:
        model.load_weights(weights_path)

    if hasattr(model, 'outputs_confidences') and model.outputs_confidences:
        preds, confidences = model.predict(x, batch_size=batch_size)
    else:
        preds = model.predict(x, batch_size=batch_size)
        confidences = None  # No confidences available

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

    # Prepare data for plotting
    data = []
    for class_name, counts in class_counts.items():
        data.append([class_name, counts['correct'], counts['incorrect'], counts['correct'] + counts['incorrect']])

    # Create DataFrame for easier plotting
    df = pd.DataFrame(data, columns=['Class', 'Correct Count', 'Incorrect Count', 'Total Count'])

    # Sort data by total count for better readability
    df_sorted = df.sort_values('Total Count', ascending=True)

    # Set the figure size
    plt.figure(figsize=(20, 140))

    # Define bar positions and width
    bar_width = 0.7  # Reduced bar width
    spacing = 0.15  # Increased spacing between bars
    y_positions = np.arange(len(df_sorted))

    # Plot the correct predictions (green) and incorrect predictions (red) with a log scale
    plt.barh(y_positions, df_sorted['Correct Count'], color='green', edgecolor='black', height=bar_width, label='Correct')
    plt.barh(y_positions, df_sorted['Incorrect Count'], left=df_sorted['Correct Count'], color='red', edgecolor='black', height=bar_width, label='Incorrect')

    # Use logarithmic scale for the x-axis
    plt.xscale('log')

    # Adjusting the y-ticks and labels with increased spacing
    plt.gca().set_yticks(y_positions + (spacing / 2))  # Add spacing to the positions
    plt.gca().set_yticklabels(df_sorted['Class'])

    # Aesthetics and labels
    plt.xlabel('Number of Predictions (Log Scale)')
    plt.title('Correct and Incorrect Predictions by Class')
    plt.legend(title='Prediction')

    # Save or show the plot
    plt.tight_layout()
    if saving_path:
        plt.savefig(saving_path)
    if show_plot:
        plt.show()

def compare_models(training_paths: list[str], model_names=[], show='both', show_best_result=False, saving_path=None, show_plot=True, num_epochs_shown=None):

    # Ensure model names are provided
    if not model_names:
        model_names = [f"Model {i+1}" for i in range(len(training_paths))]
    else:
        assert len(model_names) == len(training_paths), "Number of model names must match number of training paths."

    assert show in ['train', 'test', 'both'], "Parameter 'show' must be 'train', 'test', or 'both'."

    # Initialize a dictionary to hold the histories for each model
    models_histories = {}

    # Read the history JSON files for each model
    for path, model_name in zip(training_paths, model_names):
        # Find the JSON file in the path
        json_files = [file for file in os.listdir(path) if file.endswith('.json')]
        assert json_files, f"No JSON history file found in {path}"
        json_file = json_files[0]  # Assuming one JSON file per path

        json_path = os.path.join(path, json_file)
        with open(json_path, 'r') as f:
            history = json.load(f)
        models_histories[model_name] = history

    # Define metrics to plot
    metrics = ['accuracy', 'precision', 'recall', 'f1-score']

    # Prepare subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()

    # Setup color cycle
    color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    model_colors = {model_name: next(color_cycle) for model_name in model_names}

    # Plot each metric
    for i, metric in enumerate(metrics):
        ax = axes[i]
        for model_name, history in models_histories.items():
            color = model_colors[model_name]
            # Determine if we have data to plot
            has_train = show in ['train', 'both'] and 'train' in history and metric in history['train']
            has_test = show in ['test', 'both'] and 'test' in history and metric in history['test']

            # Plot training metric
            if has_train:
                train_metric = history['train'][metric]
                if num_epochs_shown is None:
                    epochs = range(1, len(train_metric) + 1)
                else:
                    num_epochs_shown = num_epochs_shown if num_epochs_shown<=len(train_metric) else len(train_metric)
                    epochs = range(1, num_epochs_shown + 1)
                    train_metric = train_metric[:num_epochs_shown]
                ax.plot(
                    epochs,
                    train_metric,
                    label=f'{model_name} Train',
                    linestyle='-',
                    marker='',
                    color=color
                )
                if show_best_result:
                    # Find max value
                    best_value = max(train_metric)
                    # Plot horizontal line at best_value
                    ax.axhline(y=best_value, color=color, linestyle=':', label=f'{model_name} Best Train')
            # Plot testing metric
            if has_test:
                test_metric = history['test'][metric]
                if num_epochs_shown is None:
                    epochs = range(1, len(test_metric) + 1)
                else:
                    num_epochs_shown = num_epochs_shown if num_epochs_shown<=len(test_metric) else len(test_metric)
                    epochs = range(1, num_epochs_shown + 1)
                    test_metric = test_metric[:num_epochs_shown]
                ax.plot(
                    epochs,
                    test_metric,
                    label=f'{model_name} Test',
                    linestyle='--',
                    marker='',
                    color=color
                )
                if show_best_result:
                    # Find max value
                    best_value = max(test_metric)
                    # Plot horizontal line at best_value
                    ax.axhline(y=best_value, color=color, linestyle='-.', label=f'{model_name} Best Test')

            # Handle case where neither train nor test data is available
            if not has_train and not has_test:
                print(f"Warning: No data to plot for metric '{metric}' for model '{model_name}'.")

        ax.set_xlabel('Epochs')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'Model Comparison - {metric.capitalize()}')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    if saving_path:
        plt.savefig(saving_path)
    if show_plot:
        plt.show()
