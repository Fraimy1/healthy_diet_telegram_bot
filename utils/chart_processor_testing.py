from utils.chart_data_processor import (
    prepare_bju_chart_data, prepare_macro_chart_data,
    prepare_bju_dynamics_chart_data, prepare_izp_dynamics_chart_data,
    prepare_nutrient_deficiencies_chart_data, prepare_vitamins_chart_data
)
from utils.data_processor import restructure_microelements

def test_bju_chart_data():
    """
    Test the BJU chart data preparation function with a specific user ID.
    Prints the results to console for verification.
    """
    # Get the microelements table
    microelements_table = restructure_microelements()
    
    # Test user ID
    user_id = 968466884
    
    # Get BJU data
    bju_values, bju_sources = prepare_bju_chart_data(user_id, microelements_table)
    
    # Print results
    print("\n=== BJU Chart Data Test ===")
    print("\nTotal amounts (in grams):")
    for category, amount in bju_values.items():
        print(f"{category}: {amount:.2f}g")
    
    print("\nTop sources for each category:")
    for category, sources in bju_sources.items():
        print(f"\n{category} sources (top 3):")
        for source, amount in sources[:3]:  # Show only top 3 sources
            print(f"- {source}: {amount:.2f}g")

def test_macro_chart_data():
    """Test the macro elements chart data preparation."""
    microelements_table = restructure_microelements()
    user_id = 968466884
    
    macro_data = prepare_macro_chart_data(user_id, microelements_table)
    
    print("\n=== Macro Elements Chart Data Test ===")
    print("\nMacro elements amounts (in grams):")
    for element, data in macro_data.items():
        print(f"\n{element}:")
        print(f"Total amount: {data['total_amount']:.2f}g")
        print("Top 3 sources:")
        for source, amount in data['sources'][:3]:
            print(f"- {source}: {amount:.2f}g")

def test_bju_dynamics_chart_data():
    """Test the BJU dynamics chart data preparation."""
    microelements_table = restructure_microelements()
    user_id = 968466884
    
    bju_series = prepare_bju_dynamics_chart_data(user_id, microelements_table)
    
    print("\n=== BJU Dynamics Chart Data Test ===")
    for category, time_series in bju_series.items():
        print(f"\n{category} dynamics:")
        for date, amount in time_series[:5]:  # Show first 5 entries
            print(f"- {date.strftime('%Y-%m-%d %H:%M')}: {amount:.2f}g")

def test_izp_dynamics_chart_data():
    """Test the IZP dynamics chart data preparation."""
    microelements_table = restructure_microelements()
    user_id = 968466884
    
    izp_series = prepare_izp_dynamics_chart_data(user_id, microelements_table)
    
    print("\n=== IZP Dynamics Chart Data Test ===")
    for element, time_series in izp_series.items():
        print(f"\n{element} dynamics:")
        for date, amount in time_series[:5]:  # Show first 5 entries
            print(f"- {date.strftime('%Y-%m-%d %H:%M')}: {amount:.2f}mg")

def test_nutrient_deficiencies_chart_data():
    """Test the nutrient deficiencies chart data preparation."""
    microelements_table = restructure_microelements()
    user_id = 968466884
    
    deficiencies_data = prepare_nutrient_deficiencies_chart_data(user_id, microelements_table)
    
    print("\n=== Nutrient Deficiencies Chart Data Test ===")
    print("Distribution of nutrients by status:")
    for category, percentage in deficiencies_data.items():
        print(f"{category}: {percentage:.1f}%")

def test_vitamins_chart_data():
    """Test the vitamins chart data preparation."""
    microelements_table = restructure_microelements()
    user_id = 968466884
    
    fact_values, norm_values = prepare_vitamins_chart_data(user_id, microelements_table)
    
    vitamin_names = [
        "Витамин A", "Витамин D", "Витамин E", 
        "Витамин B1", "Витамин B2", "Витамин PP",
        "Витамин B4", "Витамин B5", "Витамин B6",
        "Витамин B8", "Витамин B9", "Витамин B12",
        "Витамин C", "Витамин K"
    ]
    
    print("\n=== Vitamins Chart Data Test ===")
    print("Vitamin levels (fact vs norm):")
    for i, vitamin in enumerate(vitamin_names):
        percentage = (fact_values[i] / norm_values[i] * 100) if norm_values[i] > 0 else 0
        print(f"\n{vitamin}:")
        print(f"- Actual amount: {fact_values[i]:.2f}")
        print(f"- Daily norm: {norm_values[i]:.2f}")
        print(f"- Percentage of norm: {percentage:.1f}%")

if __name__ == "__main__":
    test_bju_chart_data()
    test_macro_chart_data()
    test_bju_dynamics_chart_data()
    test_izp_dynamics_chart_data()
    test_nutrient_deficiencies_chart_data()
    test_vitamins_chart_data()
