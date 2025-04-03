import reportlab
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
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
import os
import tempfile
import time
from utils.logger import logger
from datetime import datetime
from utils.db_utils import get_connection, Users


def register_fonts():
    """Register fonts that support Cyrillic characters"""
    # Define the location of your font files
    # You'll need to make sure these font files are available in your project
    font_dir = os.path.join(os.path.dirname(__file__), "fonts")
    
    # Create fonts directory if it doesn't exist
    os.makedirs(font_dir, exist_ok=True)
    
    # Define font mapping
    fonts = [
        ('DejaVuSans', 'DejaVuSans.ttf'),
        ('DejaVuSans-Bold', 'DejaVuSans-Bold.ttf'),
        ('DejaVuSerif', 'DejaVuSerif.ttf'),
        ('DejaVuSerif-Bold', 'DejaVuSerif-Bold.ttf'),
        ('DejaVuSans-Oblique', 'DejaVuSans-Oblique.ttf')
    ]
    
    # Register each font with ReportLab
    for font_name, font_file in fonts:
        font_path = os.path.join(font_dir, font_file)
        
        # Check if font file exists, if not download it
        if not os.path.exists(font_path):
            logger.warning(f"Font file {font_file} not found. Please place font files in {font_dir} directory.")
            # You may want to implement a download function here
            # For now, we'll skip and use system fonts as fallback
            continue
            
        try:
            pdfmetrics.registerFont(TTFont(font_name, font_path))
            logger.debug(f"Registered font: {font_name}")
        except Exception as e:
            logger.error(f"Error registering font {font_name}: {e}")
    
    # If no custom fonts were registered, try to use system fonts
    try:
        # Try to register Liberation fonts which are often available on Linux systems
        # and support Cyrillic
        system_fonts = [
            ('LiberationSans', '/usr/share/fonts/liberation/LiberationSans-Regular.ttf'),
            ('LiberationSans-Bold', '/usr/share/fonts/liberation/LiberationSans-Bold.ttf'),
            ('LiberationSerif', '/usr/share/fonts/liberation/LiberationSerif-Regular.ttf'),
            ('LiberationSerif-Bold', '/usr/share/fonts/liberation/LiberationSerif-Bold.ttf'),
            ('LiberationSans-Italic', '/usr/share/fonts/liberation/LiberationSans-Italic.ttf')
        ]
        
        for font_name, font_path in system_fonts:
            if os.path.exists(font_path):
                pdfmetrics.registerFont(TTFont(font_name, font_path))
                logger.debug(f"Registered system font: {font_name}")
    except Exception as e:
        logger.error(f"Error registering system fonts: {e}")


def create_custom_styles():
    """Create custom paragraph styles for the report using Cyrillic-compatible fonts"""
    styles = getSampleStyleSheet()
    
    # First register fonts
    register_fonts()
    
    # Try to use our registered fonts, fall back to built-in fonts if needed
    try:
        font_name = 'DejaVuSans'
        bold_font = 'DejaVuSans-Bold'
        italic_font = 'DejaVuSans-Oblique'
        
        # Check if we registered the fonts successfully
        if font_name not in pdfmetrics._fonts:
            # Try Liberation fonts
            if 'LiberationSans' in pdfmetrics._fonts:
                font_name = 'LiberationSans'
                bold_font = 'LiberationSans-Bold'
                italic_font = 'LiberationSans-Italic'
            else:
                # Fall back to Helvetica (which won't display Cyrillic properly)
                font_name = 'Helvetica'
                bold_font = 'Helvetica-Bold'
                italic_font = 'Helvetica-Oblique'
                logger.warning("Using Helvetica font which doesn't support Cyrillic. Text may not display correctly.")
    except Exception as e:
        logger.error(f"Error setting up fonts: {e}")
        # Fall back to defaults
        font_name = 'Helvetica'
        bold_font = 'Helvetica-Bold'
        italic_font = 'Helvetica-Oblique'
    
    # Title style
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontName=bold_font,
        fontSize=18,
        alignment=TA_CENTER,
        spaceAfter=12
    )
    
    # Heading style
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontName=bold_font,
        fontSize=16,
        spaceAfter=8,
        spaceBefore=12
    )
    
    # Normal text style
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=12,
        alignment=TA_JUSTIFY,
        spaceAfter=6
    )
    
    # Description text style
    description_style = ParagraphStyle(
        'Description',
        parent=styles['Italic'],
        fontName=italic_font,
        fontSize=10,
        alignment=TA_JUSTIFY,
        spaceAfter=12
    )
    
    return {
        'title': title_style,
        'heading': heading_style,
        'normal': normal_style,
        'description': description_style
    }


def get_user_info(user_id):
    """Retrieve user information from the database"""
    try:
        with get_connection() as session:
            user = session.query(Users).filter(Users.user_id == user_id).first()
            if user:
                return {
                    'name': user.fullname or "Пользователь",
                    'age': user.age,
                    'gender': "Мужской" if user.gender == "male" else "Женский" if user.gender == "female" else "Не указан",
                    'height': user.height,
                    'weight': user.weight
                }
    except Exception as e:
        logger.error(f"Error getting user info: {e}")
    
    return {
        'name': "Пользователь",
        'age': None,
        'gender': "Не указан",
        'height': None,
        'weight': None
    }


def generate_user_report_pdf(user_id, output_path):
    """
    Generate a complete PDF report with all charts for a user
    
    Args:
        user_id (int): User ID
        output_path (str): Path to save the PDF
    
    Returns:
        bool: True if report was generated successfully, False otherwise
    """
    try:
        start_time = time.time()
        logger.info(f"Starting PDF generation for user {user_id}")
        
        # Create temp directory for charts
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize calculator
            calculator = ChartsCalculator(user_id=user_id)
            
            # Get user info
            user_info = get_user_info(user_id)
            
            # Generate all charts
            chart_paths = {}
            
            # 1. BJU Chart
            logger.debug("Generating BJU chart")
            bju_data = calculator.bju_chart()
            if not bju_data.empty:
                chart_path = os.path.join(temp_dir, "bju_chart.png")
                create_bju_chart(bju_data, save_path=chart_path, show_chart=False)
                chart_paths['bju'] = chart_path
            
            # 2. BJU Macro Chart
            logger.debug("Generating BJU Macro chart")
            bju_macro_data = calculator.bju_and_others_chart()
            if not bju_macro_data.empty:
                chart_path = os.path.join(temp_dir, "bju_macro_chart.png")
                create_bju_macro_chart(bju_macro_data, save_path=chart_path, show_chart=False)
                chart_paths['bju_macro'] = chart_path
            
            # 3. BJU Dynamics Chart
            logger.debug("Generating BJU Dynamics chart")
            bju_dynamics_data = calculator.bju_dynamics_chart()
            if bju_dynamics_data:
                chart_path = os.path.join(temp_dir, "bju_dynamics_chart.png")
                create_bju_dynamics_chart(bju_dynamics_data, save_path=chart_path, show_chart=False)
                chart_paths['bju_dynamics'] = chart_path
            
            # 4. Health Index Chart
            logger.debug("Generating Health Index chart")
            izp_value, izp_details = calculator.calculate_izp()
            if izp_value is not None:
                chart_path = os.path.join(temp_dir, "health_index_chart.png")
                create_health_index_chart(izp_value, save_path=chart_path, show_chart=False)
                chart_paths['health_index'] = chart_path
            
            # 5. Percentile Chart
            logger.debug("Generating Percentile chart")
            percentile = calculator.calculate_percentile_chart()
            if percentile is not None:
                chart_path = os.path.join(temp_dir, "percentile_chart.png")
                create_percentile_chart(percentile, save_path=chart_path, show_chart=False)
                chart_paths['percentile'] = chart_path
            
            # 6. IZP Dynamics Chart
            logger.debug("Generating IZP Dynamics chart")
            izp_dynamics_data = calculator.calculate_izp_dynamics()
            if izp_dynamics_data:
                chart_path = os.path.join(temp_dir, "izp_dynamics_chart.png")
                create_izp_dynamics_chart(izp_dynamics_data, save_path=chart_path, show_chart=False)
                chart_paths['izp_dynamics'] = chart_path
            
            # 7. Food Category Donut Chart
            logger.debug("Generating Food Category chart")
            categories_data = calculator.calculate_categories_chart()
            if categories_data:
                chart_path = os.path.join(temp_dir, "food_category_chart.png")
                create_food_category_donut_chart(categories_data, save_path=chart_path, show_chart=False)
                chart_paths['food_category'] = chart_path
            
            # 8. Processing Level Donut Chart
            logger.debug("Generating Processing Level chart")
            processing_data = calculator.calculate_processing_level_chart()
            if processing_data:
                chart_path = os.path.join(temp_dir, "processing_level_chart.png")
                create_processing_level_donut_chart(processing_data, save_path=chart_path, show_chart=False)
                chart_paths['processing_level'] = chart_path
            
            # 9. Nutrient Deficiencies Donut Chart
            logger.debug("Generating Nutrient Deficiencies chart")
            deficits_data = calculator.calculate_deficits_chart()
            if deficits_data:
                chart_path = os.path.join(temp_dir, "nutrient_deficiencies_chart.png")
                create_nutrient_deficiencies_donut_chart(deficits_data, save_path=chart_path, show_chart=False)
                chart_paths['nutrient_deficiencies'] = chart_path
            
            # 10. Vitamins Chart
            logger.debug("Generating Vitamins chart")
            vitamins_data = calculator.calculate_vitamins_chart()
            if vitamins_data:
                chart_path = os.path.join(temp_dir, "vitamins_chart.png")
                create_vitamins_chart(vitamins_data, save_path=chart_path, show_chart=False)
                chart_paths['vitamins'] = chart_path
            
            # 11. Minerals Chart
            logger.debug("Generating Minerals chart")
            minerals_data = calculator.calculate_minerals_chart()
            if minerals_data:
                chart_path = os.path.join(temp_dir, "minerals_chart.png")
                create_minerals_chart(minerals_data, save_path=chart_path, show_chart=False)
                chart_paths['minerals'] = chart_path
            
            logger.info(f"Generated {len(chart_paths)} charts in {time.time() - start_time:.2f} seconds")
            
            # Check if any charts were generated
            if not chart_paths:
                logger.warning(f"No charts could be generated for user {user_id}")
                return False
                
            # Create PDF document
            pdf_start_time = time.time()
            logger.debug(f"Starting PDF assembly")
            
            # Get styles
            styles = create_custom_styles()
            
            # Create PDF document
            doc = SimpleDocTemplate(
                output_path,
                pagesize=A4,
                rightMargin=2*cm,
                leftMargin=2*cm,
                topMargin=2*cm,
                bottomMargin=2*cm
            )
            
            story = []
            
            # Title
            current_date = datetime.now().strftime("%d.%m.%Y")
            title = f"Отчет о питании для {user_info['name']}"
            story.append(Paragraph(title, styles['title']))
            story.append(Paragraph(f"Дата создания: {current_date}", styles['normal']))
            
            # User info
            if user_info['age'] or user_info['height'] or user_info['weight'] or user_info['gender'] != "Не указан":
                user_details = []
                if user_info['age']:
                    user_details.append(f"Возраст: {user_info['age']}")
                if user_info['gender'] != "Не указан":
                    user_details.append(f"Пол: {user_info['gender']}")
                if user_info['height']:
                    user_details.append(f"Рост: {user_info['height']} см")
                if user_info['weight']:
                    user_details.append(f"Вес: {user_info['weight']} кг")
                
                story.append(Paragraph("Персональные данные:", styles['heading']))
                story.append(Paragraph(", ".join(user_details), styles['normal']))
            
            story.append(Spacer(1, 0.5*cm))
            
            # Introduction
            story.append(Paragraph("Анализ питания", styles['heading']))
            intro_text = (
                "Данный отчет представляет анализ вашего питания на основе данных о приобретенных продуктах. "
                "Отчет включает информацию о макронутриентах, индексе здорового питания (ИЗП), категориях продуктов, "
                "уровне их переработки и содержании витаминов и минералов."
            )
            story.append(Paragraph(intro_text, styles['normal']))
            story.append(Spacer(1, 0.5*cm))
            
            # Health Index section (if available)
            if 'health_index' in chart_paths and izp_value is not None:
                story.append(Paragraph("Индекс здорового питания (ИЗП)", styles['heading']))
                izp_description = (
                    f"Ваш индекс здорового питания составляет {izp_value} баллов из 100. "
                    "ИЗП оценивает качество питания по 10 компонентам, включая потребление разных групп продуктов, "
                    "содержание жиров, добавленных сахаров и соли."
                )
                story.append(Paragraph(izp_description, styles['normal']))
                
                # Add health index chart
                img = Image(chart_paths['health_index'])
                img.drawHeight = 10*cm  # About 1/2 of A4 height
                img.drawWidth = 10*cm   # Maintain aspect ratio
                story.append(img)
                
                if 'percentile' in chart_paths and percentile is not None:
                    percentile_text = (
                        f"Ваш ИЗП находится в {percentile:.0f}-м перцентиле среди других пользователей. "
                        f"Это означает, что примерно {(100 - percentile):.0f}% пользователей имеют более высокий ИЗП."
                    )
                    story.append(Paragraph(percentile_text, styles['normal']))
                    
                    img = Image(chart_paths['percentile'])
                    img.drawHeight = 10*cm
                    img.drawWidth = 10*cm
                    story.append(img)
                
                if 'izp_dynamics' in chart_paths:
                    story.append(Paragraph("Динамика ИЗП", styles['heading']))
                    dynamics_text = (
                        "График ниже показывает изменения вашего индекса здорового питания за последние месяцы. "
                        "Это помогает отслеживать прогресс и видеть тенденции в качестве питания."
                    )
                    story.append(Paragraph(dynamics_text, styles['normal']))
                    
                    img = Image(chart_paths['izp_dynamics'])
                    img.drawHeight = 10*cm
                    img.drawWidth = 15*cm  # Wider for timeline chart
                    story.append(img)
                
                story.append(PageBreak())
            
            # Macronutrients section
            story.append(Paragraph("Макронутриенты", styles['heading']))
            
            if 'bju' in chart_paths:
                bju_text = (
                    "Соотношение белков, жиров и углеводов (БЖУ) в вашем рационе. "
                    "Рекомендуемое соотношение обычно составляет примерно 15% белков, 30% жиров и 55% углеводов."
                )
                story.append(Paragraph(bju_text, styles['normal']))
                
                img = Image(chart_paths['bju'])
                img.drawHeight = 10*cm
                img.drawWidth = 10*cm
                story.append(img)
            
            if 'bju_macro' in chart_paths:
                bju_macro_text = (
                    "Расширенная диаграмма макронутриентов включает также воду и пищевые волокна, "
                    "которые являются важными компонентами здорового питания."
                )
                story.append(Paragraph(bju_macro_text, styles['normal']))
                
                img = Image(chart_paths['bju_macro'])
                img.drawHeight = 10*cm
                img.drawWidth = 10*cm
                story.append(img)
            
            if 'bju_dynamics' in chart_paths:
                story.append(Paragraph("Динамика потребления макронутриентов", styles['heading']))
                bju_dynamics_text = (
                    "График показывает изменение потребления белков, жиров и углеводов за последние месяцы. "
                    "Это помогает отслеживать баланс макронутриентов в рационе со временем."
                )
                story.append(Paragraph(bju_dynamics_text, styles['normal']))
                
                img = Image(chart_paths['bju_dynamics'])
                img.drawHeight = 10*cm
                img.drawWidth = 15*cm
                story.append(img)
            
            story.append(PageBreak())
            
            # Food categories section
            if 'food_category' in chart_paths or 'processing_level' in chart_paths:
                story.append(Paragraph("Категории продуктов и уровень переработки", styles['heading']))
                
                if 'food_category' in chart_paths:
                    categories_text = (
                        "Диаграмма показывает распределение продуктов по категориям в вашем рационе. "
                        "Здоровый рацион обычно богат овощами, фруктами, цельнозерновыми продуктами и белками."
                    )
                    story.append(Paragraph(categories_text, styles['normal']))
                    
                    img = Image(chart_paths['food_category'])
                    img.drawHeight = 10*cm
                    img.drawWidth = 10*cm
                    story.append(img)
                
                if 'processing_level' in chart_paths:
                    processing_text = (
                        "Уровень переработки продуктов влияет на их питательную ценность. "
                        "Минимально обработанные продукты обычно содержат больше питательных веществ и меньше добавок."
                    )
                    story.append(Paragraph(processing_text, styles['normal']))
                    
                    img = Image(chart_paths['processing_level'])
                    img.drawHeight = 10*cm
                    img.drawWidth = 10*cm
                    story.append(img)
                    
                story.append(PageBreak())
            
            # Deficiencies section
            if 'nutrient_deficiencies' in chart_paths:
                story.append(Paragraph("Дефициты питательных веществ", styles['heading']))
                deficiencies_text = (
                    "Диаграмма отображает соотношение питательных веществ, которые находятся в пределах нормы, "
                    "и тех, которые находятся в дефиците в вашем рационе."
                )
                story.append(Paragraph(deficiencies_text, styles['normal']))
                
                img = Image(chart_paths['nutrient_deficiencies'])
                img.drawHeight = 10*cm
                img.drawWidth = 10*cm
                story.append(img)
            
            if 'vitamins' in chart_paths or 'minerals' in chart_paths:
                story.append(Paragraph("Витамины и минералы", styles['heading']))
                vitamins_minerals_text = (
                    "Оптимальное потребление витаминов и минералов важно для поддержания здоровья. "
                    "Графики ниже показывают процент от рекомендуемой нормы каждого вещества в вашем рационе."
                )
                story.append(Paragraph(vitamins_minerals_text, styles['normal']))
                
                if 'vitamins' in chart_paths:
                    img = Image(chart_paths['vitamins'])
                    img.drawHeight = 12*cm
                    img.drawWidth = 16*cm
                    story.append(img)
                
                if 'minerals' in chart_paths:
                    img = Image(chart_paths['minerals'])
                    img.drawHeight = 12*cm
                    img.drawWidth = 16*cm
                    story.append(img)
            
            # Conclusion
            story.append(Paragraph("Заключение", styles['heading']))
            conclusion_text = (
                "Данный отчет предоставляет объективную информацию о вашем питании, основанную на данных о приобретенных продуктах. "
                "Для достижения сбалансированного питания рекомендуется обеспечить разнообразие продуктов из разных групп, "
                "отдавать предпочтение нежирным белкам, свежим овощам и фруктам, а также минимально обработанным продуктам. "
                "При наличии дефицитов рекомендуется скорректировать рацион или проконсультироваться с диетологом."
            )
            story.append(Paragraph(conclusion_text, styles['normal']))
            
            # Footer with disclaimer
            disclaimer = (
                "Примечание: Данный отчет основан на информации о приобретенных продуктах и предназначен только для информационных целей. "
                "Он не заменяет консультацию профессионального диетолога или врача."
            )
            story.append(Spacer(1, 1*cm))
            story.append(Paragraph(disclaimer, styles['description']))
            
            # Build the PDF
            doc.build(story)
            logger.info(f"PDF successfully created at {output_path} in {time.time() - pdf_start_time:.2f} seconds")
            
            # Total time for report generation
            logger.info(f"Total report generation time: {time.time() - start_time:.2f} seconds")
            
            return True
            
    except Exception as e:
        logger.error(f"Error generating PDF report: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    # Example usage
    user_id = 0  # Example user ID
    output_path = "user_nutrition_report.pdf"
    generate_user_report_pdf(user_id, output_path)