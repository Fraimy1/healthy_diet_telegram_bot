import os 
import gdown
import json
from config.repository_config import LABEL_ENCODER_LINK, BERT_MODEL_LINK, BINARY_MODEL_LINK, DYNAMIC_DATASET_LINK, CLEANED_MICROELEMENTS_TABLE_LINK
from config.config import DATABASE_PATH, MODEL_CONFIG_PATH

def prepare_path(path:str):
    """
    Takes a Google Drive path and returns a direct link for gdown to download.
    
    Args:
        path (str): The Google Drive path to convert.
    
    Returns:
        str: The direct link for gdown to download.
    """
    id = path.split('/')[-2]
    updated_path = f'https://drive.google.com/uc?id={id}'
    return updated_path

def data_initialization():
    """
    Downloads all necessary data for the project to run: model weights, dynamic dataset and cleaned microelements table.
    
    If no .env file is found in the config directory, creates one with a placeholder token.
    Asserts that all links are provided in config/repository_config.py.
    Creates DATABASE_PATH and model_config['weights_path'] directories if they don't exist.
    If database_info.json doesn't exist, creates it with an empty list of unique receipt IDs.
    
    Downloads model weights and data from Google Drive using the links provided in config/repository_config.py.
    Prints a message when the data initialization is completed.
    """
    if not os.path.exists(os.path.join('config', '.env')):
        with open(os.path.join('config', '.env'), 'w') as f:
            f.write('TELEGRAM_TOKEN=#Your token goes here')
    

    os.makedirs(DATABASE_PATH, exist_ok=True)
    os.makedirs(model_config['weights_path'], exist_ok=True)

    if not os.path.exists(os.path.join(DATABASE_PATH, 'database_info.json')):
        with open(os.path.join(DATABASE_PATH, 'database_info.json'), 'w') as f:
            json.dump({'unique_receipt_ids': []}, f)
    
    try:     
        assert all([LABEL_ENCODER_LINK, BERT_MODEL_LINK, BINARY_MODEL_LINK, DYNAMIC_DATASET_LINK, CLEANED_MICROELEMENTS_TABLE_LINK]), 'All links must be provided in config/repository_config.py'
    except AssertionError as e:
        print(e)
        print('Data initialization partially completed')
        return

    # Download weights
    gdown.download(prepare_path(LABEL_ENCODER_LINK), model_config['label_encoder_path'], quiet=False)
    gdown.download(prepare_path(BERT_MODEL_LINK), model_config['bert_best_weights_path'], quiet=False)
    gdown.download(prepare_path(BINARY_MODEL_LINK), model_config['binary_best_weights_path'], quiet=False)

    # Download data
    gdown.download(prepare_path(DYNAMIC_DATASET_LINK), os.path.join(DATABASE_PATH, 'dynamic_dataset.csv'), quiet=False)
    gdown.download(prepare_path(CLEANED_MICROELEMENTS_TABLE_LINK), os.path.join(DATABASE_PATH, 'cleaned_microelements_table.csv'), quiet=False)

    print('Data initialization completed')

with open(MODEL_CONFIG_PATH, 'r') as f:
    model_config = json.load(f)

data_initialization()
