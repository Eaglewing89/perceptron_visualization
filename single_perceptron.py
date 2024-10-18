import numpy as np
import pandas as pd
import plotly.graph_objects as go
from functions import create_circle_class, create_square_class, create_figure
from classes import Perceptron


# np.set_printoptions(precision=2)


class_one = create_square_class(50, -1)
class_two = create_circle_class(50, 1)

df_one = pd.DataFrame(class_one, columns=["x", "y", "label"])
df_two = pd.DataFrame(class_two, columns=["x", "y", "label"])
df_two["x"] = df_two["x"]+1.5
df_two["y"] = df_two["y"]+1.5

df = pd.concat([df_one, df_two])

df = df.sample(frac=1)
df = df.reset_index(drop=True)


x_train = np.array(df.drop(columns="label"))
y_train = np.array(df["label"])
epochs = 40

percept = Perceptron()
w_history = percept.fit(x_train=x_train, y_train=y_train, epochs=epochs)


# Create the base figure
fig = create_figure(weights=w_history[0],
                    x_train=x_train, df=df)

# Calculate the axis ranges
x_min, x_max = x_train[:, 0].min() - 0.5, x_train[:, 0].max() + 0.5
y_min, y_max = x_train[:, 1].min() - 0.5, x_train[:, 1].max() + 0.5

# Update the layout
fig.update_layout(
    title='2D Points with Decision Boundary and Colored Half-spaces',
    xaxis_title='X',
    yaxis_title='Y',
    width=800,
    height=600,
    xaxis={"range": [x_min, x_max], 'visible': False,
           'showticklabels': False, "hoverformat": '.2f'},
    yaxis={"range": [y_min, y_max], 'visible': False,
           'showticklabels': False, "hoverformat": '.2f'},
)

fig.update_yaxes(
    scaleanchor="x",
    scaleratio=1,
)


# Create and add frames
frames = [go.Frame(data=create_figure(weights=w, x_train=x_train, df=df).data, name=str(i))
          for i, w in enumerate(w_history)]
fig.frames = frames

# Add slider and play button
fig.update_layout(
    updatemenus=[
        dict(
            type='buttons',
            showactive=False,
            buttons=[
                dict(label='Play',
                     method='animate',
                     args=[None, {'frame': {'duration': 100, 'redraw': True}, 'fromcurrent': True}]),
                dict(label='Pause',
                     method='animate',
                     args=[[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}])
            ]
        )
    ],
    sliders=[
        dict(
            steps=[
                dict(
                    method='animate',
                    args=[[str(i)], {'frame': {'duration': 100,
                                               'redraw': True}, 'mode': 'immediate'}],
                    label=str(i)
                )
                for i in range(epochs)
            ],
            transition={'duration': 0},
            x=0,
            y=0,
            currentvalue={'font': {'size': 12}, 'prefix': 'Iteration: ',
                          'visible': True, 'xanchor': 'center'},
            len=0.9,
        )
    ]
)

fig.show()
