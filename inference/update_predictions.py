from bot.bot_init import bert_2level_model, le
from utils.db_utils import get_connection, ReceiptItems, UserSettings, Users
from utils.parser import Parser
from tqdm import tqdm
import pandas as pd

def update_predictions():
    """
    Updates predictions for all items in the database using the current model.
    Uses user-specific confidence thresholds from UserSettings.
    Shows a progress bar during processing.
    """
    parser = Parser()
    
    with get_connection() as session:
        # Get user settings for confidence thresholds
        user_settings = session.query(Users.user_id, UserSettings.minimal_prediction_confidence)\
            .join(UserSettings)\
            .all()
        confidence_thresholds = {user_id: conf for user_id, conf in user_settings}
        
        # Get all items with their user_ids
        items = session.query(
            ReceiptItems.item_id, 
            ReceiptItems.product_name,
            ReceiptItems.user_id
        ).all()
        
        if not items:
            return
        
        print("Parsing product names...")
        with tqdm(total=3, desc="Processing") as pbar:
            # Parse product names
            product_names = [item.product_name for item in items]
            parsed_data = parser.parse_dataset(
                product_names,
                extract_cost=False,
                extract_hierarchical_number=False,
            )
            pbar.update(1)
            
            # Get new predictions
            predictions, confidences = bert_2level_model.predict(parsed_data['product_name'].tolist())
            predictions = le.inverse_transform(predictions)
            pbar.update(1)
            
            # Create user_predictions based on user-specific confidence thresholds
            user_predictions = [
                pred if conf >= confidence_thresholds.get(item.user_id, 0.5) else 'нераспознанное'
                for pred, conf, item in zip(predictions, confidences, items)
            ]
            pbar.update(1)
        
        # Update database in batches
        batch_size = 1000
        
        print("\nUpdating database...")
        with tqdm(total=len(items), desc="Updating records") as pbar:
            for i in range(0, len(items), batch_size):
                batch_items = items[i:i + batch_size]
                batch_predictions = predictions[i:i + batch_size]
                batch_confidences = confidences[i:i + batch_size]
                batch_user_predictions = user_predictions[i:i + batch_size]
                
                # Update each item in the batch
                for item, pred, conf, user_pred in zip(
                    batch_items, 
                    batch_predictions, 
                    batch_confidences, 
                    batch_user_predictions
                ):
                    session.query(ReceiptItems).filter(
                        ReceiptItems.item_id == item.item_id
                    ).update({
                        'prediction': pred,
                        'confidence': float(conf),
                        'user_prediction': user_pred
                    })
                    pbar.update(1)
                
                # Commit each batch
                session.commit()
        
        print("\nPrediction update completed successfully!")

if __name__ == '__main__':
    update_predictions()
