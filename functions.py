"""
Collection of necessary functions
"""

from math import pi, sin, cos
from random import uniform
import pandas as pd
import numpy as np
import plotly.graph_objects as go



def create_square_class(number: int, label: int) -> list[list[float]]:
    points = []
    for point in range(number):
        points.append((uniform(-1, 1), uniform(-1, 1), label))
    return points


def create_circle_class(number: int, label: int):
    points = []
    for point in range(number):
        angle = uniform(0, 2 * pi)
        # points.append((cos(angle), sin(angle), label))
        points.append((uniform(cos(angle), 0), uniform(sin(angle), 0), label))
    return points


def create_figure(weights: np.array, x_train: pd.DataFrame, df: pd.DataFrame):
    # Calculate the decision boundary line
    x_range = np.linspace(x_train[:, 0].min()-10, x_train[:, 0].max()+10, 1000)
    y_boundary = -(weights[1] * x_range + weights[0]) / weights[2]

    # Create a color scheme that matches Plotly Express default colors
    color_discrete_map = {-1: 'rgb(31, 119, 180)', 1: 'rgb(255, 127, 14)'}

    # Create the base plot with colored half-spaces
    fig = go.Figure()

    # Add colored half-spaces
    fig.add_trace(go.Scatter(
        x=[x_range[0], x_range[-1], x_range[-1], x_range[0]],
        y=[y_boundary[0], y_boundary[-1],
            np.max(y_boundary) + 10, np.max(y_boundary) + 10],
        fill='toself',
        #fillcolor='rgba(0,0,255,0.1)',
        fillcolor=f'rgba{color_discrete_map[1][3:-1]}, 0.1)',  # Class 1 color with 0.1 opacity
        line=dict(color='rgba(0,0,0,0)'),  # Transparent line,
        showlegend=False,
        hoverinfo='skip'
    ))

    fig.add_trace(go.Scatter(
        x=[x_range[0], x_range[-1], x_range[-1], x_range[0]],
        y=[y_boundary[0], y_boundary[-1],
            np.min(y_boundary) - 10, np.min(y_boundary) - 10],
        fill='toself',
        #fillcolor='rgba(255,0,0,0.1)',
        fillcolor=f'rgba{color_discrete_map[-1][3:-1]}, 0.1)',  # Class 0 color with 0.1 opacity
        line=dict(color='rgba(0,0,0,0)'),  # Transparent line,
        showlegend=False,
        hoverinfo='skip'
    ))

    for class_value in [-1, 1]:
        mask = df["label"] == class_value
        fig.add_trace(go.Scatter(
            x=df[mask]['x'],
            y=df[mask]['y'],
            mode='markers',
            marker=dict(color='red' if class_value == -1 else 'blue'),
            name=f'Class {class_value}'
        ))

    # Add the decision boundary line
    fig.add_trace(go.Scatter(x=x_range, y=y_boundary, mode='lines', name='Boundary      ',
                             line=dict(color='green', width=2)))

    return fig
