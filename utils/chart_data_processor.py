from config.config import MICROELEMENTS_PATH
from utils.db_utils import get_connection
import pandas as pd
import numpy as np

class ChartsCalculator:
    def __init__(self):
        # Read raw table without processing first
        self._microelements_table = pd.read_csv(MICROELEMENTS_PATH, sep='|')
        # Get multipliers from row 1 before converting to numeric
        self._multipliers_to_gram = self._get_multipliers_to_gram()
        # Now convert to numeric for later calculations
        self._process_numeric_columns()
    
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
    

if __name__ == '__main__':
    calculator = ChartsCalculator()
    print(calculator.get_microelements_for_product('Молоко пастеризованное', 200))  # Example usage
