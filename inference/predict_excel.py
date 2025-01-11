import pandas as pd
from model.model_init import initialize_model
from utils.parser import Parser
from IPython.display import display

highlighted_items_path = 'data/highlighted_items_20241227_234053.xlsx'
items_path = 'data/items_20241227_234054.xlsx'

highlighted_items = pd.read_excel(highlighted_items_path)
items = pd.read_excel(items_path)

highlighted_items.dropna(subset=['name'], inplace=True)
items.dropna(subset=['name'], inplace=True)

bert_2level_model, le = initialize_model()
parser = Parser()

highlighted_items['cleaned_names'] = parser.clean_dataset(highlighted_items['name'])
items['cleaned_names'] = parser.clean_dataset(items['name'])

# Fix: Pass only the cleaned text data to the model
highlighted_items['prediction'] = le.inverse_transform(bert_2level_model.predict(highlighted_items['cleaned_names'].tolist(), 1024)[0])
items['prediction'] = le.inverse_transform(bert_2level_model.predict(items['cleaned_names'].tolist(), 1024)[0])

highlighted_items.drop(columns=['cleaned_names'], inplace=True)
items.drop(columns=['cleaned_names'], inplace=True)

highlighted_items.to_excel('data/highlighted_items_20241227_234053_predicted.xlsx', index=False)
items.to_excel('data/items_20241227_234054_predicted.xlsx', index=False)
