"""
Collection of necessary functions
"""

from math import pi, sin, cos
from random import uniform
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from classes import MultiLayerPerceptron



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
    color_discrete_map = {-1: "rgb(31, 119, 180)", 1: "rgb(255, 127, 14)"}

    # Create the base plot with colored half-spaces
    fig = go.Figure()

    # Add colored half-spaces
    fig.add_trace(go.Scatter(
        x=[x_range[0], x_range[-1], x_range[-1], x_range[0]],
        y=[y_boundary[0], y_boundary[-1],
            np.max(y_boundary) + 10, np.max(y_boundary) + 10],
        fill="toself",
        #fillcolor="rgba(0,0,255,0.1)",
        fillcolor=f"rgba{color_discrete_map[1][3:-1]}, 0.1)",  # Class 1 color with 0.1 opacity
        line=dict(color="rgba(0,0,0,0)"),  # Transparent line,
        showlegend=False,
        hoverinfo="skip"
    ))

    fig.add_trace(go.Scatter(
        x=[x_range[0], x_range[-1], x_range[-1], x_range[0]],
        y=[y_boundary[0], y_boundary[-1],
            np.min(y_boundary) - 10, np.min(y_boundary) - 10],
        fill="toself",
        #fillcolor="rgba(255,0,0,0.1)",
        fillcolor=f"rgba{color_discrete_map[-1][3:-1]}, 0.1)",  # Class 0 color with 0.1 opacity
        line=dict(color="rgba(0,0,0,0)"),  # Transparent line,
        showlegend=False,
        hoverinfo="skip"
    ))

    for class_value in [-1, 1]:
        mask = df["label"] == class_value
        fig.add_trace(go.Scatter(
            x=df[mask]["x"],
            y=df[mask]["y"],
            mode="markers",
            marker=dict(color="red" if class_value == -1 else "blue"),
            name=f"Class {class_value}"
        ))

    # Add the decision boundary line
    fig.add_trace(go.Scatter(x=x_range, y=y_boundary, mode="lines", name="Boundary      ",
                             line=dict(color="#401266", width=2)))

    return fig


def create_figure_mlp(model: MultiLayerPerceptron, features: np.array, resolution: int=100, index: int=-1) -> go.Figure:
    """
    Plot the decision boundary of a mlp neural network. 2d inputs.

    Args:
        model (MultiLayerPerceptron): Trained mlp model.
        features (np.array): feature of shape (n, 2).
        resolution (int, optional): number of points in each linspace. Defaults to 100.
        index (int, optional): which set of weights to use for prediction. -1 is the latest. Defaults to -1.

    Returns:
        go.Figure: plot figure.
    """
    # Create mesh grid
    x_min, x_max = features[:, 0].min() - 1.5, features[:, 0].max() + 1.5
    y_min, y_max = features[:, 1].min() - 1.5, features[:, 1].max() + 1.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                        np.linspace(y_min, y_max, resolution))

    # Get predictions for all mesh grid points
    pred = model.predict(np.c_[xx.ravel(), yy.ravel()], index=index)
    pred = pred.reshape(xx.shape)

    # Create the contour plot for decision boundary
    fig = go.Figure()

    colorscale = [[0, "rgb(255, 127, 14)"], [1, "rgb(31, 119, 180)"]]

    # Add contour plot
    fig.add_trace(go.Contour(
        x=np.linspace(x_min, x_max, resolution),
        y=np.linspace(y_min, y_max, resolution),
        z=pred,
        colorscale=colorscale,
        opacity=0.10,
        showscale=False,
        contours=dict(
            start=0,
            end=1,
            size=0.5
        )
    ))

    # Identify boundary points
    boundary_x, boundary_y = [], []
    for i in range(pred.shape[0] - 1):
        for j in range(pred.shape[1] - 1):
            # Check if neighboring points differ, indicating a boundary
            if pred[i, j] != pred[i + 1, j] or pred[i, j] != pred[i, j + 1]:
                boundary_x.append(xx[i, j])
                boundary_y.append(yy[i, j])

    # # Add boundary line as a purple scatter plot
    # fig.add_trace(go.Scatter(
    #     x=boundary_x,
    #     y=boundary_y,
    #     mode="markers",
    #     marker=dict(color="purple", size=2),
    #     name="Decision Boundary"
    # ))

    return fig
