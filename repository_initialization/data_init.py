import os 
import gdown
import json
from config.repository_config import LABEL_ENCODER_LINK, BERT_MODEL_LINK, BINARY_MODEL_LINK, DYNAMIC_DATASET_LINK, CLEANED_MICROELEMENTS_TABLE_LINK
from config.config import DATABASE_PATH, MODEL_CONFIG_PATH

def prepare_path(path:str):
    id = path.split('/')[-2]
    updated_path = f'https://drive.google.com/uc?id={id}'
    return updated_path

def data_initialization():

    os.makedirs(DATABASE_PATH, exist_ok=True)
    os.makedirs(model_config['weights_path'], exist_ok=True)

    if not os.path.exists(os.path.join(DATABASE_PATH, 'database_info.json')):
        with open(os.path.join(DATABASE_PATH, 'database_info.json'), 'w') as f:
            json.dump({'unique_receipt_ids': []}, f)
        
    # Download weights
    gdown.download(prepare_path(LABEL_ENCODER_LINK), model_config['label_encoder_path'], quiet=False)
    gdown.download(prepare_path(BERT_MODEL_LINK), model_config['bert_best_weights_path'], quiet=False)
    gdown.download(prepare_path(BINARY_MODEL_LINK), model_config['binary_best_weights_path'], quiet=False)

    # Download data
    gdown.download(prepare_path(CLEANED_MICROELEMENTS_TABLE_LINK), os.path.join(DATABASE_PATH, 'cleaned_microelements_table.csv'), quiet=False)
    gdown.download(prepare_path(DYNAMIC_DATASET_LINK), os.path.join(DATABASE_PATH, 'dynamic_dataset.csv'), quiet=False)

    print('Data initialization completed')

with open(MODEL_CONFIG_PATH, 'r') as f:
    model_config = json.load(f)

data_initialization()