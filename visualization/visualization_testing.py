import os
from utils.chart_data_processor import ChartsCalculator
from visualization.visual_utils import (
    create_bju_chart,
    create_bju_macro_chart,
    create_bju_dynamics_chart,
    create_health_index_chart,
    create_percentile_chart,
    create_izp_dynamics_chart,
    create_food_category_donut_chart,
    create_processing_level_donut_chart,
    create_nutrient_deficiencies_donut_chart,
    create_vitamins_chart,
    create_minerals_chart
)


def test_visualizations():
    # Create output directory if it doesn't exist
    output_dir = "visualization/test_visualization_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize calculator with test user (id=0) and debug mode
    calculator = ChartsCalculator(user_id=0, debug=True)
    
    # Test 1: BJU Chart
    print("Testing BJU Chart...")
    bju_data = calculator.bju_chart()
    if not bju_data.empty:
        create_bju_chart(
            bju_data,
            save_path=os.path.join(output_dir, "bju_chart.png"),
            show_chart=False
        )
        print("BJU Chart created successfully")
    
    # Test 2: BJU Macro Chart
    print("\nTesting BJU Macro Chart...")
    bju_macro_data = calculator.bju_and_others_chart()
    if not bju_macro_data.empty:
        create_bju_macro_chart(
            bju_macro_data,
            save_path=os.path.join(output_dir, "bju_macro_chart.png"),
            show_chart=False
        )
        print("BJU Macro Chart created successfully")
    
    # Test 3: BJU Dynamics Chart
    print("\nTesting BJU Dynamics Chart...")
    bju_dynamics_data = calculator.bju_dynamics_chart()
    if bju_dynamics_data:
        create_bju_dynamics_chart(
            bju_dynamics_data,
            save_path=os.path.join(output_dir, "bju_dynamics_chart.png"),
            show_chart=False
        )
        print("BJU Dynamics Chart created successfully")
    
    # Test 4: Health Index Chart
    print("\nTesting Health Index Chart...")
    izp_value, _ = calculator.calculate_izp()
    if izp_value is not None:
        create_health_index_chart(
            izp_value,
            save_path=os.path.join(output_dir, "health_index_chart.png"),
            show_chart=False
        )
        print("Health Index Chart created successfully")
    
    # Test 5: Percentile Chart
    print("\nTesting Percentile Chart...")
    percentile = calculator.calculate_percentile_chart()
    if percentile is not None:
        create_percentile_chart(
            percentile,
            save_path=os.path.join(output_dir, "percentile_chart.png"),
            show_chart=False
        )
        print("Percentile Chart created successfully")
    
    # Test 6: IZP Dynamics Chart
    print("\nTesting IZP Dynamics Chart...")
    izp_dynamics_data = calculator.calculate_izp_dynamics()
    if izp_dynamics_data:
        create_izp_dynamics_chart(
            izp_dynamics_data,
            save_path=os.path.join(output_dir, "izp_dynamics_chart.png"),
            show_chart=False
        )
        print("IZP Dynamics Chart created successfully")
    
    # Test 7: Food Category Donut Chart
    print("\nTesting Food Category Donut Chart...")
    categories_data = calculator.calculate_categories_chart()
    if categories_data:
        create_food_category_donut_chart(
            categories_data,
            save_path=os.path.join(output_dir, "food_category_donut_chart.png"),
            show_chart=False
        )
        print("Food Category Donut Chart created successfully")
    
    # Test 8: Processing Level Donut Chart
    print("\nTesting Processing Level Donut Chart...")
    processing_data = calculator.calculate_processing_level_chart()
    if processing_data:
        create_processing_level_donut_chart(
            processing_data,
            save_path=os.path.join(output_dir, "processing_level_donut_chart.png"),
            show_chart=False
        )
        print("Processing Level Donut Chart created successfully")
    
    # Test 9: Nutrient Deficiencies Donut Chart
    print("\nTesting Nutrient Deficiencies Donut Chart...")
    deficits_data = calculator.calculate_deficits_chart()
    if deficits_data:
        create_nutrient_deficiencies_donut_chart(
            deficits_data,
            save_path=os.path.join(output_dir, "nutrient_deficiencies_donut_chart.png"),
            show_chart=False
        )
        print("Nutrient Deficiencies Donut Chart created successfully")
    
    # Test 10: Vitamins Chart
    print("\nTesting Vitamins Chart...")
    vitamins_data = calculator.calculate_vitamins_chart()
    if vitamins_data:
        create_vitamins_chart(
            vitamins_data,
            save_path=os.path.join(output_dir, "vitamins_chart.png"),
            show_chart=False
        )
        print("Vitamins Chart created successfully")
    
    # Test 11: Minerals Chart
    print("\nTesting Minerals Chart...")
    minerals_data = calculator.calculate_minerals_chart()
    if minerals_data:
        create_minerals_chart(
            minerals_data,
            save_path=os.path.join(output_dir, "minerals_chart.png"),
            show_chart=False
        )
        print("Minerals Chart created successfully")

if __name__ == "__main__":
    test_visualizations()
