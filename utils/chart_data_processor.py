from config.config import MICROELEMENTS_PATH
from utils.db_utils import get_connection, UserPurchases, ReceiptItems, Users, to_dict
import pandas as pd
import numpy as np
from datetime import datetime
from utils.parser import Parser
from utils.logger import logger

parser = Parser()

n_to_category = {  
    1: "Молочные",  
    2: "Яйца",  
    3: "Мясо",  
    4: "Рыба",  
    5: "Жир",  
    6: "Зерновые",  
    7: "Бобовые, орехи",  
    8: "Овощи",  
    9: "Фрукты",  
    10: "Кондитерские",  
    11: "Напитки",  
    12: "Специи",  
    13: "Фастфуд",  
    14: "Корм животных",  
    15: "Дет питание"  
}

n_to_processing_level = {
    0:'мало обработанные',
    1:'высокий уровень обработки',
    2:'Готовые блюда или полуфабрикаты',
    3:'Фастфуд'
}

microelements_columns = ['Вод', 'Бел', 'Жир', 'Угл', 'Зол', 'НЖК', 'МНЖК', 'ПНЖК', 'Омега6', 'Омега3', 'Хол', 'ТЖК', 
                         'МДС', 'Крахм', 'ПВ', 'OK', 'Фруктз', 'Лакт', 'А', 'A ', 'Кар', 'D ', 'Е', 'K ', 'B1', 
                         'В2', 'B3(PP)', 'PP', 'В4', 'B5', 'B6', 'B8(H)', 'B9', 'В12', 'С', 'Са', 'P', 'Mg', 'К', 'Na', 
                         'Cl', 'Fe', ' Zn', ' I', ' Cu', ' Mn', ' Mo', ' Se', ' Cr', ' Co', ' F', ' Si', ' V', ' Sr', ' Al', 
                         ' Ti', ' Ni', ' Sn', ' Zr', ' Rb', ' Li', 'B', 'S', 'Вал', 'Изо', 'Лей', 'Лиз', 'Мет', 'Тре', 'Три', 'Фен', 
                         'Ала', 'Арг', 'Асп', 'Гис', 'Гли', 'Глу', 'Про', 'Сер', 'Тир', 'Цис', 'Окс', 'Алк', 'Коф', 'Тео', 'Нит', 'Пур', 'Окс.1']

vitamins_columns = ['А', 'A ', 'Кар', 'D ', 'Е', 'K ', 'B1', 'В2', 'B3(PP)', 'PP', 'В4', 'B5', 'B6', 'B8(H)', 'B9', 'В12', 'С']

minerals_columns = ['Са', 'P', 'Mg', 'К', 'Na', 'Cl', 'Fe', ' Zn', ' I', ' Cu', ' Mn', ' Mo', ' Se', ' Cr', ' Co', ' F', ' Si']

izp_levels = {
    "Зерновые": (5.0, 18.5, 32.0, 45.5, 59.0, 72.5, 86.0, 99.5, 113.0, 126.5, 140),
    "Овощи": (0.0, 10.8, 21.6, 32.4, 43.2, 54.0, 64.8, 75.6, 86.4, 97.2, 108),
    "Фрукты": (0.0, 9.0, 18.0, 27.0, 36.0, 45.0, 54.0, 63.0, 72.0, 81.0, 90),
    "Молочные": (0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5, 180.0, 202.5, 225),
    "Мясопродукты": (0.0, 12.6, 25.2, 37.8, 50.4, 63.0, 75.6, 88.2, 100.8, 113.4, 126),
    "% жира по калорийности": (45.0, 43.5, 42.0, 40.5, 39.0, 37.5, 36.0, 34.5, 33.0, 31.5, 30.0),
    "% насыщенных жирных кислот по калорийности": (25.0, 23.5, 22.0, 20.5, 19.0, 17.5, 16.0, 14.5, 13.0, 11.5, 10.0),
    "% добавленного сахара по калорийности": (25.0, 23.5, 22.0, 20.5, 19.0, 17.5, 16.0, 14.5, 13.0, 11.5, 10.0),
    "Холестерин": (450.0, 435.0, 420.0, 405.0, 390.0, 375.0, 360.0, 345.0, 330.0, 315.0, 300.0),
    "Поваренная соль": (5.0, 4.775, 4.6, 4.3, 4.1, 3.9, 3.7, 3.4, 3.2, 2.975, 2.75),
}

def calculate_points(component, value):
    thresholds = izp_levels.get(component)
    
    if thresholds is None:
        logger.debug('No thresholds found for %s', component)
        return None  
    
    is_reversed = thresholds[0] > thresholds[-1] # Check if the first value is bigger than the last one
    
    if not is_reversed:
        for i in range(len(thresholds)):
            if value <= thresholds[i]:
                return i - 1 if i>0 else 0  # Return the corresponding points for the range
    else:
        for i in range(len(thresholds)):
            if value >= thresholds[i]:
                return i - 1 if i>0 else 0  
            # e.g. in fat 38 is bigger than 37.5, hence it would be 5 points
            # but we want to return 4 as 38 lies in range [39,37.5), so we return i - 1

    return 10 

def calculate_izp_value(izp_data):
    izp_points = 0
    for component, value in izp_data.items():
        points = calculate_points(component, value)
        izp_points += points
        logger.debug(f'{component}: {value} -> {points} points')
    return izp_points


def round_chart_values(data):
    original_order = list(data.keys())
    data = dict(sorted(data.items(), key=lambda x: x[1]))
    
    summ = 0
    for key, value in data.items():
        value = np.floor(value * 100)
        data[key] = value
        summ+= value

    for key, value in list(data.items())[::-1]:
        if summ < 100:
            data[key] += 1
            summ += 1
        else:
            break
    
    data = {original_order[i]: data[original_order[i]] for i in range(len(original_order))}

    return data

class ChartsCalculator:
    def __init__(self, user_id, debug=False):
        # Read raw table without processing first
        self._microelements_table = pd.read_csv(MICROELEMENTS_PATH, sep='|')
        # Get multipliers from row 1 before converting to numeric
        self._multipliers_to_gram = self._get_multipliers_to_gram()
        # Now convert to numeric for later calculations
        self.user_id = user_id
        self.user_data = None  # Initialize user_data as None
        self.debug = debug
        self.norms = {}

        self.izp_value = None
        self.izp_data = None

        self._process_numeric_columns()
        self._process_user_data()  # Process user data after initialization
        self._get_norms()

    def _process_numeric_columns(self):
        """Convert only numeric columns to float, preserving string columns"""
        # Expanded list of non-numeric columns
        non_numeric_columns = [
            'Продукты, назв сокр',
            'БК',
            'Код БК',
            'число продуктов'
        ]

        # First, ensure 'Код БК' remains as string type
        if 'Код БК' in self._microelements_table.columns:
            self._microelements_table['Код БК'] = self._microelements_table['Код БК'].astype(str)

        # Then convert other columns to numeric
        for column in self._microelements_table.columns:
            if column not in non_numeric_columns:
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

    def _get_norms(self):
        """Get nutrient norm values from the microelements table"""
        for col in microelements_columns:
            # Remove excessive logging of every column value
            if pd.isna(self._microelements_table.iloc[2][col]):
                self.norms[col] = None
                continue
            
            try:
                norm = self._microelements_table.iloc[2][col] * self._microelements_table.iloc[1][col] / 1000
                # to mg / 1000 => to g
                self.norms[col] = norm
            except Exception as e:
                logger.warning(f"Error calculating norm for {col}: {e}")
                self.norms[col] = None
            
        # Log summary instead of individual values
        valid_norms = sum(1 for v in self.norms.values() if v is not None)
        logger.debug(f"Loaded {valid_norms} valid norms out of {len(microelements_columns)} total columns")

    def get_data_for_user(self):
        """Get all data for the user including purchase datetime for each item"""

        with get_connection() as session:
            items_data = session.query(ReceiptItems, UserPurchases.purchase_datetime).join(
                UserPurchases,
                (UserPurchases.receipt_id == ReceiptItems.receipt_id) & 
                (UserPurchases.user_id == ReceiptItems.user_id)
            ).filter(
                ReceiptItems.user_id == self.user_id,
                ReceiptItems.user_prediction != 'несъедобное'
            ).all()
            
            if not items_data:
                return []

            # Convert objects to dicts and add purchase_datetime
            data_dicts = []
            for item, purchase_datetime in items_data:
                item_dict = to_dict(item)
                item_dict['purchase_datetime'] = purchase_datetime
                data_dicts.append(item_dict)
                
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
        user_df = user_df[
                (user_df['user_prediction'] != 'несъедобное')
                & user_df['amount'].notna()
                & user_df['quantity'].notna()
            ] 
        user_df['amount_in_grams'] = user_df['amount'].apply(lambda x: parser.convert_amount_units_to_grams(x)) * user_df['quantity']

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
        
        # merged_df.to_excel('user_data_without_conversion.xlsx')    

        # Get amount of each microlment in grams for each product
        columns_not_to_convert = ['item_id', 'receipt_id', 'user_id', 'quantity', 'percentage', 'amount', 'price', 'product_name', 
                                  'portion', 'prediction', 'user_prediction', 'confidence', 'in_history', 'purchase_datetime', 'amount_in_grams', 
                                  'Продукты, назв сокр', 'БК', 'Код БК', 'число продуктов', 'Обр']
        for column in merged_df.columns.drop(columns_not_to_convert):
            mult_to_gr  = self._multipliers_to_gram.get(column, 1)
            merged_df[column] = (mult_to_gr * merged_df[column]/100) * merged_df['amount_in_grams']


        # Store the processed data
        self.user_data = merged_df

    def _get_data_from_last_30d(self):
        
        #! For debugging purposes only
        if self.debug:
            return self._get_data_from_month(9, 2024)

        if self.user_data is None:
            return pd.DataFrame()

        today = datetime.now() #self.user_data['purchase_datetime'].max()
        thirty_days_ago = today - pd.Timedelta(days=30)
        mask = (self.user_data['purchase_datetime'] >= thirty_days_ago)
        filtered_data = self.user_data[mask]
        
        return filtered_data

    def _get_data_from_month(self, month, year):
        # Filter for last 30 days
        mask = (self.user_data['purchase_datetime'].dt.month == month) & (self.user_data['purchase_datetime'].dt.year == year)
        filtered_data = self.user_data[mask]
        
        return filtered_data

    def _get_data_from_last_12m(self):
        """
        Filter data for last 12 months and return it grouped by (year, month).
        Returns a dictionary with (year, month) tuples as keys and corresponding DataFrames as values.
        """
        # Filter for last 12 months
        today = datetime.now()
        twelve_months_ago = today - pd.DateOffset(months=12)
        mask = (self.user_data['purchase_datetime'] >= twelve_months_ago)
        
        # Create an explicit copy of the filtered data
        filtered_data = self.user_data[mask].copy()

        # Now it's safe to modify the DataFrame
        filtered_data['purchase_datetime'] = pd.to_datetime(filtered_data['purchase_datetime'])
        filtered_data['year'] = filtered_data['purchase_datetime'].dt.year
        filtered_data['month'] = filtered_data['purchase_datetime'].dt.month
        
        # Group the data by year and month
        grouped_data = {}
        for (year, month), group in filtered_data.groupby(['year', 'month']):
            grouped_data[(year, month)] = group
            
        return grouped_data

    def bju_chart(self):
        """Get the BJU chart for the user"""
        data = self._get_data_from_last_30d()

        # Convert purchase_datetime to datetime if it's not already
        self.user_data['purchase_datetime'] = pd.to_datetime(self.user_data['purchase_datetime'])
        
        if data.empty:
            return pd.DataFrame()

        # Get the sum of all purchases for each microelement
        bju_columns = ['Бел', 'Жир', 'Угл']
        bju_data = data[bju_columns].sum()
        return bju_data

    def bju_and_others_chart(self):
        """Get the BJU and microelements chart for the user"""
        data = self._get_data_from_last_30d()
        
        if data.empty:
            return pd.DataFrame()

        # Get the sum of all purchases for each microelement
        columns = ['Бел', 'Жир','Вод', 'ПВ', 'Угл']
        bju_and_others_data = data[columns].sum()
        return bju_and_others_data

    def bju_dynamics_chart(self):
        data = self._get_data_from_last_12m()

        if not data:
            return pd.DataFrame()

        # Get the sum of all purchases for each microelement
        bju_columns = ['Бел', 'Жир', 'Угл']
        for key, group in data.items():
            data[key] = group[bju_columns].sum()
        
        return data

    def calculate_izp(self, data=None):
        """Calculate the Izp value for the user"""
        if data is None:
            data = self._get_data_from_last_30d()

        # data.to_excel('user_data_jan.xlsx') 

        if data.empty:
            return None

        # Get base calories for calculations
        izp_data = {}
        total_calories = data['ЭЦ'].sum()

        logger.debug('Total calories: %s', total_calories)

        #* --- 1. Зерновые продукты ---
        grain_products = data.loc[
                    (data['Код БК'].fillna('').str.startswith('6_'))
                    & data['amount_in_grams'].notna()
                    ]
        grain_products = grain_products['amount_in_grams'].sum()
        logger.debug('Grain products: %s', grain_products)
        grain_value = grain_products * 1000 / total_calories
        logger.debug('Grain value: %s', grain_value)
        izp_data['Зерновые'] = grain_value

        #* --- 2. Молочные продукты ---
        milk_products = data.loc[
                    (data['Код БК'].fillna('').str.startswith('1_'))
                    & data['amount_in_grams'].notna()
                    ]
        milk_products = milk_products['amount_in_grams'].sum()
        logger.debug('Milk products: %s', milk_products)
        milk_value = milk_products * 1000 / total_calories
        logger.debug('Milk value: %s', milk_value)
        izp_data['Молочные'] = milk_value
        
        #* --- 3. Мясопродукты ---
        meat_products = data.loc[
                    (
                        data['Код БК'].fillna('').str.startswith('2_')
                        | data['Код БК'].fillna('').str.startswith('4_')
                        | data['Код БК'].fillna('').str.startswith('3_')
                    )
                    & data['amount_in_grams'].notna()
                    ]
                                    
        meat_products = meat_products.loc[
            ~(
                meat_products['Код БК'].str.startswith('3_3')
                | meat_products['Код БК'].str.startswith('3_6')
            )
        ]
        
        meat_products = meat_products['amount_in_grams'].sum()
        logger.debug('Meat products: %s', meat_products)
        meat_value = meat_products * 1000 / total_calories
        logger.debug('Meat value: %s', meat_value)
        izp_data['Мясопродукты'] = meat_value

        #* --- 4. Овощи ---
        vegetable_products = data.loc[
                    (
                        data['Код БК'].fillna('').str.startswith('7_')
                        | data['Код БК'].fillna('').str.startswith('8_')
                    )
                    & data['amount_in_grams'].notna()
                    ]
        vegetable_products = vegetable_products['amount_in_grams'].sum()
        logger.debug('Vegetable products: %s', vegetable_products)
        vegetable_value = vegetable_products * 1000 / total_calories
        logger.debug('Vegetable value: %s', vegetable_value)
        izp_data['Овощи'] = vegetable_value

        #* --- 5. Фрукты ---
        fruit_products = data.loc[
                    data['Код БК'].fillna('').str.startswith('9_')
                    & data['amount_in_grams'].notna()
                    ]
        fruit_products = fruit_products['amount_in_grams'].sum()
        logger.debug('Fruit products: %s', fruit_products)
        fruit_value = fruit_products * 1000 / total_calories
        logger.debug('Fruit value: %s', fruit_value)
        izp_data['Фрукты'] = fruit_value

        #* --- 6. Процент жира по калорийности ---
        # Общий жир за период * 9 / все калории * 1000
        fat = data.loc[data['Жир'].notna(), 'Жир'].sum()
        logger.debug('Fat: %s', fat)
        fat_value = fat * 9 / total_calories * 100
        logger.debug('Fat value: %s', fat_value)
        izp_data['% жира по калорийности'] = fat_value

        #* --- 7. Процент насыщенных жирных кислот ---
        # НЖК за период * 9 / все калории * 1000
        saturated_fat = data.loc[data['НЖК'].notna(), 'НЖК'].sum()
        logger.debug('Saturated fat: %s', saturated_fat)
        saturated_fat_value = saturated_fat * 9 / total_calories * 100
        logger.debug('Saturated fat value: %s', saturated_fat_value)
        izp_data['% насыщенных жирных кислот по калорийности'] = saturated_fat_value

        #* --- 8. Процент добавленного сахара ---
        # МДС от переработанных продуктов (Обр != 0) * 4 / все калории * 1000
        mds = data.loc[
                    data['Обр'].notna()
                    & data['Обр'] != 0
                    & data['quantity'].notna() 
                    & data['amount'].notna(),
                    'МДС'
                    ]
        mds = mds.sum()
        logger.debug('MDS: %s', mds)
        mds_value = mds * 4 / total_calories * 100
        logger.debug('MDS value: %s', mds_value)
        izp_data['% добавленного сахара по калорийности'] = mds_value

        #* --- 9. Холестерин ---
        # Весь холестерин / все калории * 1000 (диапазон 500-900)
        cholesterol = data.loc[data['Хол'].notna(), 'Хол'].sum()
        logger.debug('Cholesterol: %s', cholesterol)
        cholesterol_value = cholesterol * 1000 / total_calories * 1000
        logger.debug('Cholesterol value: %s', cholesterol_value)
        izp_data['Холестерин'] = cholesterol_value

        #* --- 10. Поваренная соль ---
        # (Na + Cl) в граммах / все калории * 1000
        sodium = data.loc[data['Na'].notna(), 'Na'].sum()
        logger.debug('Sodium: %s', sodium)
        chloride = data.loc[data['Cl'].notna(), 'Cl'].sum()
        logger.debug('Chloride: %s', chloride)
        salt = (sodium + chloride) / total_calories * 1000
        logger.debug('Salt value: %s', salt)
        izp_data['Поваренная соль'] = salt

        izp_value = calculate_izp_value(izp_data)
        logger.debug('Izp value: %s', izp_value)

        self.izp_data = izp_data
        self.izp_value = izp_value

        return izp_value, izp_data

    def calculate_izp_precentile_chart(self):
        if self.izp_value is None:
            self.izp_value, self.izp_data = self.calculate_izp()
        
        izp_value = self.izp_value

        #* finish when there is available data

    def calculate_categories_chart(self):
        """Calculate the categories chart for the user"""
        data = self._get_data_from_last_30d()

        if data.empty:
            return pd.DataFrame()

        category_codes = data.loc[data['Код БК'].notna(), 'Код БК']
        category_codes = category_codes.apply(lambda x: x.split('_')[0])

        category_counts = category_codes.astype(int).value_counts()
        categories_data = category_counts.sort_index().to_dict()
        categories_data = {n_to_category[key]: value for key, value in categories_data.items()}

        return categories_data

    def calculate_processing_level_chart(self):
        """Calculate the processing level chart for the user"""
        data = self._get_data_from_last_30d()

        if data.empty:
            return pd.DataFrame()

        processing_levels = data.loc[data['Обр'].notna(), 'Обр'].astype(int)

        processing_counts = processing_levels.value_counts()
        processing_data = processing_counts.sort_index().to_dict()
        sum_values = sum(processing_data.values())
        processing_data = {n_to_processing_level[key]: value / sum_values for key, value in processing_data.items()}
        
        processing_data = round_chart_values(processing_data)

        return processing_data

    def calculate_izp_dynamics(self):
        """Calculate the Izp dynamics for the user"""
        data = self._get_data_from_last_12m()

        if not data:
            return pd.DataFrame()

        izp_dynamics = {}
        for key, group in data.items():
            izp_value, izp_data = self.calculate_izp(group)
            logger.debug('Izp data for %s: %s', key, izp_data)
            izp_dynamics[key] = izp_value

        return izp_dynamics
    
    def calculate_deficits_chart(self):
        """Calculate the deficits chart for the user"""
        data = self._get_data_from_last_30d()

        if data.empty:
            return pd.DataFrame()

        all_calories = data['ЭЦ'].sum()

        deficits_data = {
            'В норме': 0,
            'В дефиците': 0,
            # 'Надо ограничить': 0,
            'Нет данных': 0
        }
        
        for col in microelements_columns:
            norm = self.norms[col]
            if norm is None:
                deficits_data['Нет данных'] += 1
                continue
    
            col_data = data[col].sum() / all_calories * 1000
            if col_data < norm:
                deficits_data['В дефиците'] += 1
            elif col_data > norm:
                deficits_data['В норме'] += 1
        
        return deficits_data

    def calculate_vitamins_chart(self):
        """Calculate the vitamins chart for the user"""
        data = self._get_data_from_last_30d()

        all_calories = data['ЭЦ'].sum()

        if data.empty:
            return pd.DataFrame()

        vitamins_data = {}
        for col in vitamins_columns:
            norm = self.norms.get(col, None)
            
            if norm is None:
                vitamins_data[col] = None
                continue
            
            vitamins_data[col] = (data[col].sum() / all_calories * 1000) / norm * 100
        
        return vitamins_data

    def calculate_minerals_chart(self):
        """Calculate the minerals chart for the user"""
        data = self._get_data_from_last_30d()

        all_calories = data['ЭЦ'].sum()

        if data.empty:
            return pd.DataFrame()

        minerals_data = {}
        for col in minerals_columns:
            norm = self.norms.get(col, None)
            
            if norm is None:
                minerals_data[col] = None
                continue
            
            minerals_data[col] = (data[col].sum() / all_calories * 1000) / norm * 100
        
        return minerals_data

    def calculate_percentile_chart(self):
        """Calculate user's IZP percentile compared to other users"""
        if self.izp_value is None:
            self.izp_value, self.izp_data = self.calculate_izp()

        if self.izp_value is None:
            return None

        # Get all IZP values including current user's
        all_izp_values = [self.izp_value]  # Start with current user's value
        
        # Get other users
        with get_connection() as session:
            other_users = session.query(Users.user_id).filter(Users.user_id != self.user_id).all()
            other_users = [user[0] for user in other_users]

        # Calculate IZP for other users
        for user_id in other_users:
            try:
                other_calculator = ChartsCalculator(user_id=user_id, debug=None)
                izp_value, _ = other_calculator.calculate_izp()
                if izp_value is not None:
                    all_izp_values.append(izp_value)
            except Exception as e:
                logger.debug(f"Failed to calculate IZP for user {user_id}: {e}")
                continue

        # Need at least 2 users (current + 1 other) to calculate percentile
        if len(all_izp_values) < 2:
            logger.debug("Not enough users with data to calculate percentile")
            return 50  # Return 50th percentile if there's no comparison data
            
        # Calculate the percentile of current user's IZP
        total_users = len(all_izp_values)
        users_below = sum(1 for x in all_izp_values if x <= self.izp_value)
        percentile = (users_below / total_users) * 100
        
        return percentile

if __name__ == '__main__':
    calculator = ChartsCalculator(user_id=0, debug=True)
    
    
    print(calculator.bju_chart())  
    
    print(calculator.bju_and_others_chart())  

    print(calculator.bju_dynamics_chart())  

    print(calculator.calculate_categories_chart())  

    print(calculator.calculate_izp())

    print(calculator.calculate_izp_dynamics())

    print(calculator.calculate_processing_level_chart())
    
    print(calculator.calculate_deficits_chart())

    print(calculator.calculate_vitamins_chart())

    print(calculator.calculate_minerals_chart())
    print('='*100, 'Debugging', '='*100, sep='\n')

    print(calculator._microelements_table.columns.tolist())
    print(calculator.user_data.columns.tolist())
    print(calculator._microelements_table.head(6))
    print(calculator._microelements_table[6:].head(10))
    print(calculator._microelements_table.iloc[6].T.to_dict())
    print(calculator.user_data.iloc[0].T.to_dict())
    print(calculator.user_data['Обр'])
    # calculator.user_data.to_excel('user_data.xlsx')

    print(calculator.user_data.iloc[1].T.to_dict())
    
    print(calculator._microelements_table)