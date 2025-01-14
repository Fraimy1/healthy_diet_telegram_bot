import joblib
import os
from model.bert_model import BertModel, BertWithBinaryBert, CustomLabelEncoder
from model import bert_model
from config.model_config import (
    NUM_LABELS, MAX_LENGTH, BERT_BEST_WEIGHTS_PATH,
    BINARY_BEST_WEIGHTS_PATH, LABEL_ENCODER_PATH,
    INEDIBLE_NUM
)
import sys

sys.modules['bert_model'] = bert_model

def initialize_model():
    """
    Initialize model and label encoder from config file.

    Returns:
        - model: instance of BertWithBinaryBert class
        - le: instance of LabelEncoder class
    """
    bert_model = BertModel(NUM_LABELS, MAX_LENGTH, use_gpu=True)
    bert_model.load_weights(BERT_BEST_WEIGHTS_PATH)
    
    binary_bert = BertModel(2, MAX_LENGTH, use_gpu=True)
    binary_bert.load_weights(BINARY_BEST_WEIGHTS_PATH)
    
    bert_2level_model = BertWithBinaryBert(bert_model, binary_bert, inedible_class=INEDIBLE_NUM)
    bert_2level_model.load_weights(BERT_BEST_WEIGHTS_PATH)
   
    le = joblib.load(LABEL_ENCODER_PATH)

    return bert_2level_model, le
