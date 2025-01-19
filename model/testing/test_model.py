import os
from pathlib import Path
from model.testing.test_utils import find_best_epoch
from model.model_init import initialize_model
from config.model_config import BERT_BEST_WEIGHTS_PATH, BINARY_BEST_WEIGHTS_PATH, LABEL_ENCODER_PATH

def initialize_testing_pipeline(weights_path: str) -> tuple[int, Path]:
    """
    Initialize the testing pipeline by finding the best epoch and creating output directory.
    
    Args:
        weights_path: Path to the folder containing model weights and metrics
    
    Returns:
        tuple: (best_epoch, output_path)
    """
    # Verify the weights path exists
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"The path {weights_path} does not exist")

    # Find the metrics JSON file
    json_files = [f for f in os.listdir(weights_path) if f.endswith('.json')]
    if not json_files:
        raise FileNotFoundError(f"No JSON metrics file found in {weights_path}")
    
    metrics_file = os.path.join(weights_path, json_files[0])
    
    # Find the best epoch using the metrics
    best_epoch = find_best_epoch(
        path_to_json=metrics_file,
        deciding_metric='f1-score',
        deciding_set='test',
        verbose=True
    )

    # Create output directory
    folder_name = os.path.basename(os.path.normpath(weights_path))
    output_path = Path(__file__).parent / folder_name
    output_path.mkdir(exist_ok=True)
    
    print(f"Created output directory at: {output_path}")
    
    return best_epoch, output_path

def get_best_weights_path(best_epoch: int, weights_path: str) -> str:
    for file in os.listdir(weights_path):
        if file.endswith('.h5'):
            file_epoch = int(file.split('_')[-1].split('.')[0])
            if file_epoch == best_epoch:
                return os.path.join(weights_path, file)

weights_path = 'model/training/history/bert_multilabel_weights_bert_2level_parsed_2025-01-14_07:39'
best_epoch, output_path = initialize_testing_pipeline(weights_path)

# +1 because the epoch starts at 1 in file names
best_weights_path = get_best_weights_path(best_epoch+1, weights_path)

if 'binary' in weights_path:
    model, label_encoder = initialize_model(
            binary_weights_path=best_weights_path,
            num_labels=2,
            return_mode='binary'
        )
else:
    model, label_encoder = initialize_model(
            bert_weights_path=best_weights_path,
            return_mode='bert'
            )

print(f"Best epoch: {best_epoch}")
print(f"Best weights path: {best_weights_path}")

