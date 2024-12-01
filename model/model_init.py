import json
import joblib
import os
from model.bert_model import BertModel, BertWithBinaryBert, CustomLabelEncoder
from model import bert_model
from config.config import MODEL_CONFIG_PATH
import sys

sys.modules['bert_model'] = bert_model

def initialize_model():
    """
    Initialize model and label encoder from config file.

    Returns:
        - model: instance of BertWithBinaryBert class
        - le: instance of LabelEncoder class
    """
    with open(MODEL_CONFIG_PATH, 'r') as f:
        config = json.load(f, )

    bert_model = BertModel(config['num_labels'], config['max_length'], use_gpu=True)
    bert_model.load_weights(config['bert_best_weights_path'])
    
    binary_bert = BertModel(2, config['max_length'], use_gpu=True)
    binary_bert.load_weights(config['binary_best_weights_path'])
    
    bert_2level_model = BertWithBinaryBert(bert_model, binary_bert, inedible_class=config['inedible_num'])
    bert_2level_model.load_weights(config['bert_best_weights_path'])
   
    le = joblib.load(config['label_encoder_path'])

    return bert_2level_model, le
