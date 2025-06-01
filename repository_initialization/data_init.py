import os 
import gdown
from config.repository_config import LABEL_ENCODER_LINK, BERT_MODEL_LINK, BINARY_MODEL_LINK, CLEANED_MICROELEMENTS_TABLE_LINK, DATASET_LINKS
from config.config import DATABASE_PATH, DATABASE_FILE_PATH, DATASETS_FOLDER
from utils.db_utils import create_database
from config.model_config import WEIGHTS_PATH, LABEL_ENCODER_PATH, BERT_BEST_WEIGHTS_PATH, BINARY_BEST_WEIGHTS_PATH

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
    Creates DATABASE_PATH and WEIGHTS_PATH directories if they don't exist.
    If database_info.json doesn't exist, creates it with an empty list of unique receipt IDs.
    
    Downloads model weights and data from Google Drive using the links provided in config/repository_config.py.
    Prints a message when the data initialization is completed.
    """
    if not os.path.exists(os.path.join('config', '.env')):
        with open(os.path.join('config', '.env'), 'w') as f:
            f.write('TELEGRAM_TOKEN=#Your token goes here')
    
    os.makedirs(DATASETS_FOLDER, exist_ok=True)

    os.makedirs(DATABASE_PATH, exist_ok=True)
    os.makedirs(WEIGHTS_PATH, exist_ok=True)

    if not os.path.exists(DATABASE_FILE_PATH):
        create_database()
    try:     
        assert all([LABEL_ENCODER_LINK, BERT_MODEL_LINK, BINARY_MODEL_LINK, CLEANED_MICROELEMENTS_TABLE_LINK, DATASET_LINKS]), 'All links must be provided in config/repository_config.py'
    except AssertionError as e:
        print(e)
        print('Data initialization partially completed')
        return

    # Download weights
    gdown.download(prepare_path(LABEL_ENCODER_LINK), LABEL_ENCODER_PATH, quiet=False)
    gdown.download(prepare_path(BERT_MODEL_LINK), BERT_BEST_WEIGHTS_PATH, quiet=False)
    gdown.download(prepare_path(BINARY_MODEL_LINK), BINARY_BEST_WEIGHTS_PATH, quiet=False)

    # Download data
    gdown.download(prepare_path(CLEANED_MICROELEMENTS_TABLE_LINK), os.path.join(DATABASE_PATH, 'cleaned_microelements_table.csv'), quiet=False)
    DATASETS_NAMES = ['dynamic_dataset_not_reviewed.csv', 'data_nov28_combined.csv', 'data_nov28_combined_parsed.csv']
    for dataset_link, database_name in zip(DATASET_LINKS, DATASETS_NAMES):
        gdown.download(prepare_path(dataset_link), os.path.join(DATASETS_FOLDER, database_name), quiet=False)

    print('Data initialization completed')


data_initialization()