from typing import Dict, Tuple, List
from utils.data_processor import get_microelements_data
from datetime import datetime
from utils.data_processor import get_sorted_user_receipts

def prepare_izp_dynamics_chart_data(user_id: int, microelements_table: dict) -> Dict[str, List[Tuple[datetime, float]]]:
    """
    Prepares data for the IZP (Iron, Zinc, Phosphorus) dynamics chart showing changes over time.
    
    Args:
        user_id (int): The ID of the user
        microelements_table (dict): The nested dictionary of microelements data
        
    Returns:
        Dict[str, List[Tuple[datetime, float]]]: Dictionary with IZP time series 
            {element_name: [(datetime, amount), ...]}
    """
    receipts = get_sorted_user_receipts(user_id)
    
    # Define the IZP elements we want to track
    izp_elements = {
        "Железо (Fe)": "Iron",
        "Цинк (Zn)": "Zinc",
        "Фосфор (P)": "Phosphorus"
    }
    
    # Initialize time series for each element
    izp_series = {name: [] for name in izp_elements.keys()}
    
    # Get data for each receipt date
    for receipt in receipts:
        receipt_date = receipt['purchase_datetime']
        microelements_data = get_microelements_data(user_id, microelements_table)
        
        for element_name in izp_elements.keys():
            if element_name in microelements_data:
                amount = microelements_data[element_name]['total_amount']
                izp_series[element_name].append((receipt_date, amount))
    
    # Sort each series by date
    for element in izp_series:
        izp_series[element].sort(key=lambda x: x[0])
    
    return izp_series
from utils.data_processor import get_microelements_data, get_sorted_user_receipts
from utils.db_utils import get_connection
from datetime import datetime

def prepare_bju_chart_data(user_id: int, microelements_table: dict) -> Tuple[Dict[str, float], Dict[str, list]]:
    """
    Prepares data for the BJU (Белки, Жиры, Углеводы) chart.
    
    Args:
        user_id (int): The ID of the user
        microelements_table (dict): The nested dictionary of microelements data
        
    Returns:
        Tuple[Dict[str, float], Dict[str, list]]: A tuple containing:
            - Dictionary with BJU values {category: amount}
            - Dictionary with sources for each category {category: [(source, amount), ...]}
    """
    # Get all microelements data
    microelements_data = get_microelements_data(user_id, microelements_table)
    
    # Define the BJU categories we want to extract
    bju_categories = {
        "Белок": "Белок",
        "Жир": "Жир",
        "Углеводы": "Углеводы"
    }
    
    # Extract BJU data
    bju_values = {}
    bju_sources = {}
    
    for category, display_name in bju_categories.items():
        if category in microelements_data:
            bju_values[display_name] = microelements_data[category]['total_amount']
            bju_sources[display_name] = microelements_data[category]['sources']
        else:
            bju_values[display_name] = 0
            bju_sources[display_name] = []
            
    return bju_values, bju_sources

def prepare_macro_chart_data(user_id: int, microelements_table: dict) -> Dict[str, dict]:
    """
    Prepares data for the macro elements chart.
    
    Args:
        user_id (int): The ID of the user
        microelements_table (dict): The nested dictionary of microelements data
        
    Returns:
        Dict[str, dict]: Dictionary with macro elements data {element: {'total_amount': float, 'sources': list}}
    """
    microelements_data = get_microelements_data(user_id, microelements_table)
    
    # Define macro elements to extract
    macro_elements = [
        "Натрий (Na)", "Калий (K)", "Кальций (Ca)", 
        "Магний (Mg)", "Фосфор (P)", "Хлор (Cl)"
    ]
    
    macro_data = {}
    for element in macro_elements:
        if element in microelements_data:
            macro_data[element] = {
                'total_amount': microelements_data[element]['total_amount'],
                'sources': microelements_data[element]['sources']
            }
        else:
            macro_data[element] = {'total_amount': 0, 'sources': []}
            
    return macro_data

def prepare_bju_dynamics_chart_data(user_id: int, microelements_table: dict) -> Dict[str, List[Tuple[datetime, float]]]:
    """
    Prepares data for the BJU dynamics chart showing changes over time.
    
    Args:
        user_id (int): The ID of the user
        microelements_table (dict): The nested dictionary of microelements data
        
    Returns:
        Dict[str, List[Tuple[datetime, float]]]: Dictionary with BJU time series 
            {category: [(datetime, amount), ...]}
    """
    receipts = get_sorted_user_receipts(user_id)
    
    # Initialize time series for each BJU category
    bju_series = {
        "Белок": [],
        "Жир": [],
        "Углеводы": []
    }
    
    # Get data for each receipt date
    for receipt in receipts:
        receipt_date = receipt['purchase_datetime']
        bju_values, _ = prepare_bju_chart_data(user_id, microelements_table)
        
        for category in bju_series.keys():
            if category in bju_values:
                bju_series[category].append((receipt_date, bju_values[category]))
    
    # Sort each series by date
    for category in bju_series:
        bju_series[category].sort(key=lambda x: x[0])
    
    return bju_series


def prepare_nutrient_deficiencies_chart_data(user_id: int, microelements_table: dict) -> Dict[str, float]:
    """
    Prepares data for the nutrient deficiencies donut chart.
    Calculates the percentage of nutrients that are in normal range, deficient, or in excess.
    
    Args:
        user_id (int): The ID of the user
        microelements_table (dict): The nested dictionary of microelements data
        
    Returns:
        Dict[str, float]: Dictionary with percentages for each category
            {'В норме': float, 'В дефиците': float, 'Надо ограничить': float}
    """
    microelements_data = get_microelements_data(user_id, microelements_table)
    
    # Define normal ranges (these could be moved to config)
    NORMAL_RANGE = {
        'lower': 0.75,  # 75% of recommended
        'upper': 1.75   # 175% of recommended
    }
    
    # Count nutrients in each category
    total_nutrients = 0
    normal_count = 0
    deficient_count = 0
    excess_count = 0
    
    for element, data in microelements_data.items():
        # Skip energy value as it's not a nutrient
        if element == "Энергетическая ценность":
            continue
            
        total_nutrients += 1
        amount = data['total_amount']
        
        # Categorize based on amount relative to recommended daily intake
        # Note: This is a simplified approach. In a real system, you'd want to
        # compare against specific recommended values for each nutrient
        if amount < NORMAL_RANGE['lower']:
            deficient_count += 1
        elif amount > NORMAL_RANGE['upper']:
            excess_count += 1
        else:
            normal_count += 1
    
    # Calculate percentages
    if total_nutrients > 0:
        return {
            'В норме': (normal_count / total_nutrients) * 100,
            'В дефиците': (deficient_count / total_nutrients) * 100,
            'Надо ограничить': (excess_count / total_nutrients) * 100
        }
    else:
        return {
            'В норме': 0,
            'В дефиците': 0,
            'Надо ограничить': 0
        }


def prepare_vitamins_chart_data(user_id: int, microelements_table: dict) -> Tuple[List[float], List[float]]:
    """
    Prepares data for the vitamins chart, calculating actual values and recommended norms.
    
    Args:
        user_id (int): The ID of the user
        microelements_table (dict): The nested dictionary of microelements data
        
    Returns:
        Tuple[List[float], List[float]]: Two lists containing:
            - fact_values: List of actual vitamin amounts
            - norm_values: List of recommended daily intake values
    """
    microelements_data = get_microelements_data(user_id, microelements_table)
    
    # Define vitamins to track (in order they should appear)
    vitamin_names = [
        "Витамин A", "Витамин D", "Витамин E", 
        "Витамин B1", "Витамин B2", "Витамин PP",
        "Витамин B4", "Витамин B5", "Витамин B6",
        "Витамин B8", "Витамин B9", "Витамин B12",
        "Витамин C", "Витамин K"
    ]
    
    # Reference daily intake values (in same units as data)
    # These values could be moved to config
    daily_norms = {
        "Витамин A": 900,  # mcg
        "Витамин D": 10,   # mcg
        "Витамин E": 15,   # mg
        "Витамин B1": 1.5, # mg
        "Витамин B2": 1.8, # mg
        "Витамин PP": 20,  # mg
        "Витамин B4": 500, # mg
        "Витамин B5": 5,   # mg
        "Витамин B6": 2,   # mg
        "Витамин B8": 50,  # mcg
        "Витамин B9": 400, # mcg
        "Витамин B12": 3,  # mcg
        "Витамин C": 90,   # mg
        "Витамин K": 120   # mcg
    }
    
    fact_values = []
    norm_values = []
    
    for vitamin in vitamin_names:
        # Get actual amount from microelements data
        amount = 0
        for key, data in microelements_data.items():
            if vitamin in key:  # Partial match to catch variations in naming
                amount = data['total_amount']
                break
        
        fact_values.append(amount)
        norm_values.append(daily_norms.get(vitamin, 100))  # Default to 100 if not found
    
    return fact_values, norm_values