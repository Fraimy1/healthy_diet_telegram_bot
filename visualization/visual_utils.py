import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
from scipy.stats import norm

# * NOTES:
# * 1. the resolutions are emperical and don't follow any strict rules right now
# * 2. the parameters of some functions should be changed for efficiency and better flexibility

def create_bju_chart(protein=15, fat=30, carbs=55, save_path=None, show_chart=True):
    assert save_path is not None or show_chart, "Either save_path or show_chart must be True"
    
    # Data for the chart
    labels = ['Белок', 'Жир', 'Углеводы']
    values = [protein, fat, carbs]
    colors = ['#B2E57B', '#FFD983', '#72CFFB']
    
    # Creating the pie chart
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors),
        hole=0.5,
        textinfo='label+percent',
        hoverinfo='label+percent+value'
    )])
    
    # Updating layout to match the uploaded chart's style and square aspect ratio
    fig.update_layout(
        title_text="Основные макронутриенты (БЖУ)",
        title_x=0.5,  # Center title
        font=dict(size=16, family='Arial'),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.1,
            xanchor="center",
            x=0.5
        ),
        autosize=False,
        width=700,  # Set width for square layout
        height=700  # Set height to match width
    )
    
    if show_chart:
        fig.show()
    
    if save_path:
        fig.write_image(save_path, width=700, height=700, scale=2)


def create_bju_macro_chart(protein=15, fat=40, water=30, fiber=5, carbs=30, save_path=None, show_chart=True):
    assert save_path is not None or show_chart, "Either save_path or show_chart must be True"
    # Data for the chart
    labels = ['Белок', 'Жир', 'Вода', 'Пищевые волокна', 'Углеводы']
    values = [protein, fat, water, fiber, carbs]
    colors = ['#B2E57B', '#FFD983', '#B2E6F9', '#FFADC9', '#72F2A4']
    
    # Creating the pie chart
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors),
        hole=0.5,
        textinfo='label+percent',
        hoverinfo='label+percent+value'
    )])
    
    # Updating layout to match the uploaded chart's style and square aspect ratio
    fig.update_layout(
        title_text="БЖУ и др. макронутриенты",
        title_x=0.5,  # Center title
        font=dict(size=16, family='Arial'),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.1,
            xanchor="center",
            x=0.5
        ),
        autosize=False,
        width=700,  # Set width for square layout
        height=700  # Set height to match width
    )
    
    if show_chart:
        fig.show()
    
    if save_path:
        fig.write_image(save_path, width=700, height=700, scale=2)

def create_bju_dynamics_chart(protein_values=[50, 55, 58, 69, 55, 63, 71, 62],
                            fat_values=[30, 28, 32, 25, 22, 30, 35, 28],
                            carbs_values=[40, 50, 45, 55, 60, 50, 40, 45],
                            quartiles=['1Q23', '2Q23', '3Q23', '4Q23', '1Q24', '2Q24', '3Q24', '4Q24'],
                            save_path=None, show_chart=True):
    assert save_path is not None or show_chart, "Either save_path or show_chart must be True"
    # Data for the chart

    # Creating the line chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=quartiles,
        y=protein_values,
        mode='lines+markers',
        name='Белок',
        line=dict(color='#B2E57B', width=3),
        marker=dict(size=8)
    ))
    fig.add_trace(go.Scatter(
        x=quartiles,
        y=fat_values,
        mode='lines+markers',
        name='Жир',
        line=dict(color='#FFD983', width=3),
        marker=dict(size=8)
    ))
    fig.add_trace(go.Scatter(
        x=quartiles,
        y=carbs_values,
        mode='lines+markers',
        name='Углеводы',
        line=dict(color='#72CFFB', width=3),
        marker=dict(size=8)
    ))

    # Updating layout to match style and make it visually appealing
    fig.update_layout(
        title_text="Динамика БЖУ",
        title_x=0.5,  # Center the title
        font=dict(size=16, family='Arial'),
        xaxis_title="Кварталы",
        yaxis_title="Граммы",
        plot_bgcolor="white",  # Set background to white
        xaxis=dict(
            showgrid=False,
            tickangle=-45  # Tilt labels for better readability
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray',  # Set gridline color
            gridwidth=1,
            griddash='dash',  # Make gridlines dashed
            zeroline=False
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        width=800,  # Slightly wider for clarity
        height=600
    )
    
    if show_chart:
        fig.show()
    
    if save_path:
        fig.write_image(save_path, width=800, height=600, scale=2)


def create_health_index_chart(achieved_score=60, save_path=None, show_chart=True):
    assert save_path is not None or show_chart, "Either save_path or show_chart must be True"
    # Data for the chart
    remaining_score = 100 - achieved_score  # Remaining score
    colors = ['#B2E57B', '#FFD983']  # Green for achieved, yellow for remaining

    # Creating the bar chart with empty space
    fig = go.Figure()

    # Add invisible bars for empty space
    fig.add_trace(go.Bar(
        x=['Левый пробел'],  # Left empty space
        y=[0],  # No height
        marker_color='rgba(0,0,0,0)',  # Invisible
        showlegend=False
    ))

    # Add the main bar
    fig.add_trace(go.Bar(
        x=['Сумма баллов (ИЗП)'],  # Main column
        y=[achieved_score],  # Achieved score
        name=f'{achieved_score} баллов из 100',
        marker_color=colors[0]
    ))
    fig.add_trace(go.Bar(
        x=['Сумма баллов (ИЗП)'],  # Main column
        y=[remaining_score],  # Remaining score
        name=f'Осталось {remaining_score} баллов',
        marker_color=colors[1]
    ))

    # Add invisible bars for empty space
    fig.add_trace(go.Bar(
        x=['Правый пробел'],  # Right empty space
        y=[0],  # No height
        marker_color='rgba(0,0,0,0)',  # Invisible
        showlegend=False
    ))

    # Updating layout for a clean and professional appearance
    fig.update_layout(
        title_text="Индекс здорового питания",
        title_x=0.5,  # Center the title
        font=dict(size=16, family='Arial'),
        xaxis=dict(
            title="",
            showticklabels=False,  # Hide tick labels
            zeroline=False,
        ),
        yaxis=dict(
            title="",
            showgrid=True,
            gridcolor='lightgray',  # Dashed gridlines
            gridwidth=1,
            griddash='dash',
            range=[0, 100],  # Set range from 0 to 100
            zeroline=False
        ),
        barmode='stack',  # Stack the bars
        plot_bgcolor="white",  # White background
        width=600,  # Adjust layout width
        height=600,
        showlegend=False  # Hide legend for simplicity
    )
    
    # Add the larger score text as a second heading at the top
    fig.add_annotation(
        text=f"{achieved_score} баллов из 100",
        x=0.5,
        y=1.10,  # Position above the chart title
        showarrow=False,
        xref="paper",
        yref="paper",
        font=dict(size=22, family='Arial', color='black')
    )

    if show_chart:
        fig.show()
    
    if save_path:
        fig.write_image(save_path, width=600, height=600, scale=2)


def create_percentile_chart(percentile=85, save_path=None, show_chart=True):
    assert save_path is not None or show_chart, "Either save_path or show_chart must be True"
    # Data for the normal distribution
    mean = 0
    std_dev = 1
    x = np.linspace(-2.5, 2.5, 500)  # Focused range for visual clarity
    y = norm.pdf(x, mean, std_dev)

    # Calculate the x-value for the given percentile
    percentile_value = norm.ppf(percentile / 100, mean, std_dev)

    # Create the Plotly figure
    fig = go.Figure()

    # Add the bell curve line
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines',
        line=dict(color='blue', width=2),
        showlegend=False  # Remove legend for this line
    ))

    # Add the shaded area for the percentile
    fig.add_trace(go.Scatter(
        x=x[x <= percentile_value],
        y=y[x <= percentile_value],
        fill='tozeroy',
        mode='lines',
        fillcolor='rgba(0, 0, 255, 0.2)',
        line=dict(color='blue', width=0),
        showlegend=False  # Remove legend for this shaded area
    ))

    # Add a line grid below the bell curve
    for tick in [-2, -1, 0, 1, 2]:
        fig.add_trace(go.Scatter(
            x=[tick, tick],
            y=[0, norm.pdf(tick, mean, std_dev)],
            mode='lines',
            line=dict(color='lightgray', width=1, dash='dash'),
            showlegend=False
        ))

    # Customize the layout
    fig.update_layout(
        title=f"Ваш ИЗП находится в {percentile} перцентиле.<br>Только {100 - percentile}% людей имеют более высокий ИЗП.",
        title_x=0.5,
        title_font=dict(size=16, color='gray'),
        xaxis=dict(
            showgrid=True,  # Add grid for percentiles
            zeroline=False,
            showticklabels=True,
            tickvals=[-2, -1, 0, 1, 2],  # Example tick values
            ticktext=['2.5%', '16%', '50%', '84%', '97.5%'],  # Percentile labels
            gridcolor='lightgray',
            title="Процентили"
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        plot_bgcolor='white',
        width=600,  # Adjusted width for thinner aspect
        height=600,  # Adjusted height to make it thinner
    )

    if show_chart:
        fig.show()
    
    if save_path:
        fig.write_image(save_path, width=600, height=600, scale=2)


def create_izp_dynamics_chart(izp_values=[10, 20, 35, 45, 50, 65, 75, 90],
                            months=['сент24', 'окт24', 'ноя24', 'дек24', 'янв25', 'фев25', 'мар25', 'апр25'],
                            save_path=None, show_chart=True):
    assert save_path is not None or show_chart, "Either save_path or show_chart must be True"
    # Data for the chart

    # Creating the line chart
    fig = go.Figure()

    # Add the IZP line
    fig.add_trace(go.Scatter(
        x=months,
        y=izp_values,
        mode='lines+markers',  # Line with markers for each data point
        name='ИЗП',
        line=dict(color='#B2E57B', width=3),
        marker=dict(size=6)
    ))

    # Updating layout for a clean and professional appearance
    fig.update_layout(
        title_text="Динамика индекса здорового питания",
        title_x=0.5,  # Center the title
        font=dict(size=16, family='Arial'),
        xaxis_title="",
        xaxis=dict(
            tickangle=-45,  # Rotate month labels for better readability
            showgrid=False,  # No vertical gridlines
            zeroline=False
        ),
        yaxis=dict(
            tickmode='linear',  # Linear scale
            tick0=0,
            dtick=20,  # Major ticks every 20%
            showgrid=True,
            gridcolor='lightgray',  # Dashed gridlines
            gridwidth=1,
            griddash='dash',
            range=[0, 100],  # Set range from 0 to 100%
            zeroline=False
        ),
        plot_bgcolor="white",  # White background
        width=800,
        height=600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        )
    )
    
    if show_chart:
        fig.show()
    
    if save_path:
        fig.write_image(save_path, width=800, height=600, scale=2)


def create_food_category_donut_chart(values=[30, 7, 50, 17, 85, 10, 4, 5, 10, 6],
                                   save_path=None, show_chart=True):
    assert save_path is not None or show_chart, "Either save_path or show_chart must be True"
    # Data for the chart
    labels = ['Мясо', 'Рыба', 'Овощи', 'Фрукты', 'Зерновые', 
              'Молочные', 'Яйца', 'Бобовые', 'Кондитерские', 
              'Сахаросодержащие напитки']
    colors = ['#B2E57B', '#FFD983', '#B2E6F9', '#FFADC9', '#72F2A4', 
              '#ADD8E6', '#F4A460', '#F08080', '#8A2BE2', '#FFA07A']

    # Combine labels and values for legend formatting
    formatted_labels = [f"{label} {value}" for label, value in zip(labels, values)]

    # Creating the donut chart
    fig = go.Figure(data=[go.Pie(
        labels=formatted_labels,  # Use formatted labels
        values=values,
        marker=dict(colors=colors),
        hole=0.5,
        textinfo='label+value',  # Show labels and values together
        texttemplate='%{label}',  # Label and value on the same line in the chart
        hoverinfo='label+value'  # Display label and value on hover
    )])

    # Updating layout to match the provided style
    fig.update_layout(
        title={
            'text': "Доля категорий продуктов в рационе",
            'x': 0.5,
            'y': 0.95,
            'font': dict(size=20, family='Arial')
        },
        annotations=[
            dict(
                text="грамм на 1000 ккалорий",
                x=0.5,
                y=1.065,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=20, family='Arial', color='gray')
            )
        ],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        showlegend=True,
        width=800,  # Adjust layout for square resolution
        height=800
    )
    
    if show_chart:
        fig.show()
    
    if save_path:
        fig.write_image(save_path, width=800, height=800, scale=2)


def create_processing_level_donut_chart(values=[40, 30, 20, 10],
                                      save_path=None, show_chart=True):
    assert save_path is not None or show_chart, "Either save_path or show_chart must be True"
    # Data for the chart
    labels = ['Продукты с высоким уровнем переработки', 
              'Мало переработанные продукты', 
              'Готовые блюда и полуфабрикаты', 
              'Фастфуд']
    colors = ['#B2E57B', '#FFD983', '#72CFFB', '#FFADC9']

    # Combine labels and values for legend formatting
    formatted_labels = [f"{label} {value}%" for label, value in zip(labels, values)]

    # Creating the donut chart
    fig = go.Figure(data=[go.Pie(
        labels=formatted_labels,  # Use formatted labels for legend
        values=values,
        marker=dict(colors=colors),
        hole=0.5,
        textinfo='percent',  # Display percentages on the pie chart
        texttemplate='%{percent}',  # Only show percentages
        textfont=dict(size=18),  # Increase font size for percentages
        hoverinfo='label+value'  # Display label and value on hover
    )])

    # Updating layout to match the provided style
    fig.update_layout(
        title={
            'text': "Уровень переработки купленных продуктов",
            'x': 0.5,
            'y': 0.95,
            'font': dict(size=20, family='Arial')
        },
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        showlegend=True,
        width=800,  # Adjust layout for square resolution
        height=800
    )
    
    if show_chart:
        fig.show()
    
    if save_path:
        fig.write_image(save_path, width=800, height=800, scale=2)


def create_nutrient_deficiencies_donut_chart(values=[52, 40, 8],
                                           save_path=None, show_chart=True):
    assert save_path is not None or show_chart, "Either save_path or show_chart must be True"
    # Data for the chart
    labels = ['В норме', 'В дефиците', 'Надо ограничить']
    colors = ['#B2E57B', '#FFD983', '#72CFFB']

    # Combine labels and values for legend formatting
    formatted_labels = [f"{label} {value}%" for label, value in zip(labels, values)]

    # Creating the donut chart
    fig = go.Figure(data=[go.Pie(
        labels=formatted_labels,  # Use formatted labels for legend
        values=values,
        marker=dict(colors=colors),
        hole=0.5,
        textinfo='percent',  # Show percentages inside the chart
        texttemplate='%{percent}',  # Only show percentages
        textfont=dict(size=18),  # Increase font size for percentages
        hoverinfo='label+value'  # Display label and value on hover
    )])

    # Updating layout to match the provided style
    fig.update_layout(
        title={
            'text': "Дефициты нутриентов",
            'x': 0.5,
            'y': 0.95,
            'font': dict(size=20, family='Arial')
        },
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        showlegend=True,
        width=800,  # Adjust layout for square resolution
        height=800
    )
    
    if show_chart:
        fig.show()
    
    if save_path:
        fig.write_image(save_path, width=800, height=800, scale=2)


def create_vitamins_chart(fact_values=[140, 150, 160, 100, 100, 180, 90, 70, 70, 70, 70, 70, 70, 130],
                         norm_values=None,
                         save_path=None, show_chart=True):
    assert save_path is not None or show_chart, "Either save_path or show_chart must be True"
    # Data from the table
    if norm_values is None:
        norm_values = [100] * len(fact_values)
    vitamins = [
        "Витамин A, ретинол",
        "Витамин D, кальциферол",
        "Витамин E, токоферол", 
        "Витамин B1, тиамин",
        "Витамин B2, рибофлавин",
        "Витамин PP, ниацин",
        "Витамин B4, холин",
        "Витамин B5, пантотеновая кислота",
        "Витамин B6, пиридоксин", 
        "Витамин B8 (H), биотин",
        "Витамин B9, фолаты",
        "Витамин B12, кобаламин",
        "Витамин C, аскорбиновая кислота",
        "Витамин K, филлохинон"
    ]

    fact = fact_values
    norm = norm_values
    percentages = [(f / n) * 100 for f, n in zip(fact, norm)]

    # Combine data and sort by vitamin name alphabetically
    data = sorted(zip(vitamins, percentages), key=lambda x: x[0].split(',')[0].split()[1][0], reverse=True)
    vitamins_sorted, percentages_sorted = zip(*data)

    # Colors based on percentage ranges
    colors = [
        "#A8D5BA" if p >= 100 else "#FFE29A" if p >= 50 else "#F4B6C2"
        for p in percentages_sorted
    ]

    # Create the horizontal bar chart
    fig = go.Figure()

    # Add bars
    fig.add_trace(go.Bar(
        x=percentages_sorted,
        y=vitamins_sorted,
        orientation='h',
        marker=dict(color=colors),
        text=[f"{p:.0f}%" for p in percentages_sorted],  # Add percentage text
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Процент: %{x:.0f}%<extra></extra>',
        width=0.4  # Make the bars thinner
    ))

    # Add vertical grey lines every 50%
    for x in range(0, 201, 50):  # Add lines at 50%, 100%, 150%, 200%
        fig.add_shape(
            type="line",
            x0=x, x1=x,
            y0=-0.5, y1=len(vitamins_sorted) - 0.5,
            line=dict(color="lightgrey", width=0.5, dash="dash"),
            layer="below"  # Place the lines below the bars
        )

    # Update layout
    fig.update_layout(
        title="Витамины, % от физиологической потребности",
        xaxis=dict(
            title="% от нормы",
            showgrid=False,
            zeroline=False,
            dtick=50,  # Tick every 50%
            range=[0, 200]
        ),
        yaxis=dict(
            title="",
            showgrid=False,
            categoryorder="array",
            categoryarray=vitamins_sorted  # Maintain alphabetical order in the plot
        ),
        bargap=0.5,  # Increase spacing between bars for thinner bars
        height=800,
        width=1000,
        plot_bgcolor="white"
    )
    
    if show_chart:
        fig.show()
    
    if save_path:
        fig.write_image(save_path, width=1000, height=800, scale=2)

def create_minerals_chart(fact_values=[140, 150, 350, 320, 100, 350, 90, 70, 250, 211, 32, 89, 100, 130, 500, 180, 90],
                         norm_values=[290, 310, 260, 200, 280, 270, 160, 140, 140, 140, 140, 140, 200, 230, 280, 270, 90],
                         save_path=None, show_chart=True):
    assert save_path is not None or show_chart, "Either save_path or show_chart must be True"
    # Data from the table
    minerals = [
        "Калий, K",
        "Кальций, Ca",
        "Магний, Mg",
        "Фосфор, P",
        "Железо, Fe",
        "Хлор, Cl",
        "Кремний, Si",
        "Йод, I",
        "Селен, Se",
        "Цинк, Zn",
        "Медь, Cu",
        "Марганец, Mn",
        "Молибден, Mo",
        "Фтор, F",
        "Кобальт, Co",
        "Хром, Cr",
        "Ванадий, V"
    ]

    fact = fact_values
    norm = norm_values
    percentages = [(f / n) * 100 for f, n in zip(fact, norm)]

    # Colors based on percentage ranges
    colors = [
        "#F4B6C2" if p < 50 or p >= 175 else "#A8D5BA" if 75 <= p < 175 else "#FFE29A"
        for p in percentages
    ]

    # Create the horizontal bar chart
    fig = go.Figure()

    # Add bars
    fig.add_trace(go.Bar(
        x=percentages,
        y=minerals,
        orientation='h',
        marker=dict(color=colors),
        text=[f"{p:.0f}%" for p in percentages],  # Add percentage text
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Процент: %{x:.0f}%<extra></extra>'
    ))

    # Add vertical grey lines every 50%
    for x in range(0, 401, 50):  # Add lines at 50%, 100%, 150%, etc.
        fig.add_shape(
            type="line",
            x0=x, x1=x,
            y0=-0.5, y1=len(minerals) - 0.5,
            line=dict(color="lightgrey", width=0.5, dash="dash"),
            layer="below"  # Place the lines below the bars
        )

    # Update layout
    fig.update_layout(
        title="Минералы, % от физиологической потребности",
        xaxis=dict(
            title="% от нормы",
            showgrid=False,
            zeroline=False,
            dtick=50,  # Tick every 50%
            range=[0, 400]
        ),
        yaxis=dict(
            title="",
            showgrid=False,
            categoryorder="array",
            categoryarray=minerals  # Maintain order in the plot
        ),
        bargap=0.3,  # Adjust spacing between bars for thinner bars
        height=800,
        width=1000,
        plot_bgcolor="white"
    )
    
    if show_chart:
        fig.show()
    
    if save_path:
        fig.write_image(save_path, width=1000, height=800, scale=2)