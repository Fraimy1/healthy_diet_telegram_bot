from utils.data_processor import *
# from model.model_init import initialize_model
import json

# bert_2level_model, le = initialize_model()

# test_file_path = 'data/database/user_968466884/file_28_11_2024_17_33_31.json'

# with open(test_file_path, 'r') as f:
#     data_received = json.load(f)

# receipt_data, receipt_info = parse_json(data_received)

# print(receipt_data, receipt_info)

# items_data = predict_product_categories(receipt_data, bert_2level_model, le)

# print(items_data)
user_id = 968466884
conn = sqlite3.connect(DATABASE_FILE_PATH)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

user_data = dict(cursor.execute("SELECT * FROM main_database WHERE user_id = ?", (user_id,)).fetchone())


print(user_data)