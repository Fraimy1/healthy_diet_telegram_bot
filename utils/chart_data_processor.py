from config.config import MICROELEMENTS_PATH
from utils.db_utils import get_connection, UserPurchases, ReceiptItems, to_dict
import pandas as pd
import numpy as np

class ChartsCalculator:
    def __init__(self, user_id):
        # Read raw table without processing first
        self._microelements_table = pd.read_csv(MICROELEMENTS_PATH, sep='|')
        # Get multipliers from row 1 before converting to numeric
        self._multipliers_to_gram = self._get_multipliers_to_gram()
        # Now convert to numeric for later calculations
        self.user_id = user_id
        self.user_data = None  # Initialize user_data as None
        
        self._process_numeric_columns()
        self._process_user_data()  # Process user data after initialization
    
    def _process_numeric_columns(self):
        """Convert only numeric columns to float, preserving string columns"""
        skip_columns = ['Продукты, назв сокр', 'БК', 'Код БК', 'число продуктов']
        for column in self._microelements_table.columns:
            if column not in skip_columns:
                self._microelements_table[column] = pd.to_numeric(self._microelements_table[column], errors='coerce')

    def _get_multipliers_to_gram(self):
        """Get multipliers to convert values to grams from row 1 (коэффициенты к мг)"""
        multiplier_to_gram = {}
        
        # Get the second row (index 1) which contains the multipliers
        multipliers_row = self._microelements_table.iloc[1]
        
        # Get columns excluding metadata columns
        element_names = [col for col in self._microelements_table.columns 
                        if col not in ['Продукты, назв сокр', 'БК', 'Код БК', 'число продуктов']]
        
        # Create multipliers dictionary
        for name in element_names:
            try:
                value = multipliers_row[name]
                if pd.notna(value):  # Check if value is not NaN
                    multiplier = float(str(value).replace(',', '.'))  # Handle potential comma decimals
                    multiplier_to_gram[name] = multiplier / 1000 if multiplier else 1
                else:
                    multiplier_to_gram[name] = 1
            except (ValueError, TypeError):
                multiplier_to_gram[name] = 1
                
        return multiplier_to_gram

    def get_microelements_for_product(self, product_name, amount):
        """
        Get the microelements for a given product name and amount.

        Args:
            product_name (str): The name of the product.
            amount (int): The amount of the product in grams.
            microelements_table (dict): The nested dictionary of product_name -> microelement_name -> amount.

        Returns:
            dict: A dictionary with the microelement names as keys and the amounts in grams as values.
        """
        # Filter the dataframe for the given product name
        product_row = self._microelements_table[self._microelements_table['Продукты, назв сокр'] == product_name]

        if product_row.empty:
            return {}
        
        # Convert the row to a dictionary, excluding the product_name column
        microelements = product_row.iloc[0].drop(['Продукты, назв сокр','БК','Код БК','число продуктов',]).to_dict()
        
        # Calculate amounts based on given weight
        for key, value in microelements.items():
            try:
                microelements[key] = float(value) * self._multipliers_to_gram[key]/100 * amount
            except ValueError:
                pass
            
        return microelements

    def get_data_from_time(self, date_start, date_end):
        with get_connection() as session:
            receipt_ids = session.query(UserPurchases.receipt_id).filter(
                UserPurchases.user_id == self.user_id,
                UserPurchases.purchase_datetime.between(date_start, date_end)
            ).all()
            receipt_ids = [r[0] for r in receipt_ids]
            items_data = session.query(ReceiptItems).filter(
                ReceiptItems.user_id == self.user_id,
                ReceiptItems.receipt_id.in_(receipt_ids)
            ).all()
            # Convert objects to dicts inside the session.
            if not items_data:
                return []

            data_dicts = [to_dict(item) for item in items_data]
        return data_dicts

    def get_data_from_last_30d(self):
        """Get data from the database for the last 30 days"""
        # Get the current date
        current_date = pd.Timestamp.now().date()
        # Calculate the date 30 days ago
        date_30d_ago = current_date - pd.DateOffset(days=30)
        
        return self.get_data_from_time(date_30d_ago, current_date)

    def get_data_from_last_year(self):
        """Get data from the database for the last year"""
        # Get the current date
        current_date = pd.Timestamp.now().date()
        # Calculate the date 1 year ago
        date_1y_ago = current_date - pd.DateOffset(years=1)
        
        return self.get_data_from_time(date_1y_ago, current_date)

    def get_data_for_user(self):
        """Get all data for the user"""
        with get_connection() as session:
            items_data = session.query(ReceiptItems).filter(
                ReceiptItems.user_id == self.user_id
            ).all()
            # Convert objects to dicts inside the session.
            if not items_data:
                return []

            data_dicts = [to_dict(item) for item in items_data]
        return data_dicts

    def _process_user_data(self):
        """Process user data and combine it with microelements information"""
        # Get all user data
        items_data = self.get_data_for_user()
        if not items_data:
            self.user_data = pd.DataFrame()
            return

        # Convert to DataFrame
        user_df = pd.DataFrame(items_data)
        user_df = user_df[user_df['user_prediction'] != 'несъедобное']  # Drop rows with NaN user_prediction
        
        # Clean and standardize the prediction strings
        user_df['user_prediction'] = user_df['user_prediction'].str.strip().str.lower()
        self._microelements_table['Продукты, назв сокр'] = self._microelements_table['Продукты, назв сокр'].str.strip().str.lower()
        
        # Merge user data with microelements data
        merged_df = pd.merge(
            user_df,
            self._microelements_table,
            left_on='user_prediction',
            right_on='Продукты, назв сокр',
            how='left'
        )
        
        # Store the processed data
        self.user_data = merged_df

    def bju_chart(self, period='30d'):
        """Get the BJU chart for the user"""
        # Get the data for the last year
        if period == '30d':
            data = self.get_data_from_last_30d()
        else:
            data = self.get_data_from_last_year()
        
        if not data:
            return pd.DataFrame()

        # Create a dataframe from the data
        df = pd.DataFrame(data)
        predictions = df['user_prediction']
        
        print(predictions)

        return df

if __name__ == '__main__':
    calculator = ChartsCalculator(user_id=968466884)
    print(calculator.get_microelements_for_product('Молоко пастеризованное', 200))  # Example usage
    
    print(calculator.get_data_from_time('2024-12-10', '2025-01-01'))  # Example usage

    print(calculator.bju_chart('year'))  # Example usage

    print(calculator.get_data_for_user())  # Example usage

    # print(calculator._microelements_table.head(10))

    # data = calculator.user_data
    # print(data.columns.tolist())
    # print(data.head(10))  # Example usage