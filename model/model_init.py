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

def initialize_model(
    bert_weights_path: str = BERT_BEST_WEIGHTS_PATH,
    binary_weights_path: str = BINARY_BEST_WEIGHTS_PATH,
    label_encoder_path: str = LABEL_ENCODER_PATH,
    num_labels: int = NUM_LABELS,
    max_length: int = MAX_LENGTH,
    inedible_num: int = INEDIBLE_NUM,
    use_gpu: bool = True,
    return_mode = 'both'
) -> tuple[BertWithBinaryBert, CustomLabelEncoder]:
    """
    Initialize model and label encoder with configurable parameters.

    Args:
        bert_weights_path: Path to BERT model weights
        binary_weights_path: Path to binary BERT model weights
        label_encoder_path: Path to label encoder file
        num_labels: Number of classification labels
        max_length: Maximum sequence length
        inedible_num: Inedible class number
        use_gpu: Whether to use GPU
        return_mode: 'both' to return both model and label encoder, 'binary' to return only binary BERT model, 'bert' to return only multilabel BERT model

    Returns:
        tuple: (model, label_encoder)
    """
    le = joblib.load(label_encoder_path)
    if return_mode == 'binary':
        binary_bert = BertModel(2, max_length, use_gpu=use_gpu)
        binary_bert.load_weights(binary_weights_path)
        return binary_bert, le
    elif return_mode == 'bert':
        bert_model = BertModel(num_labels, max_length, use_gpu=use_gpu)
        bert_model.load_weights(bert_weights_path)
        return bert_model, le
    
    bert_model = BertModel(num_labels, max_length, use_gpu=use_gpu)
    bert_model.load_weights(bert_weights_path)
    
    binary_bert = BertModel(2, max_length, use_gpu=use_gpu)
    binary_bert.load_weights(binary_weights_path)
    
    bert_2level_model = BertWithBinaryBert(bert_model, binary_bert, inedible_class=inedible_num)
    bert_2level_model.load_weights(bert_weights_path)
   
    return bert_2level_model, le
