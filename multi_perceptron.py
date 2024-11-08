import numpy as np
import pandas as pd
from classes import PointCollection
import plotly_express as px
import plotly.graph_objects as go


dataset = PointCollection()
dataset.add_ellipse_collection(20, label=1)
dataset.change_collection_location(0, -2.5, 1)
dataset.add_ellipse_collection(20, label=1)
dataset.change_collection_location(1, 2.5, 1)
dataset.add_ellipse_collection(20, label=-1)
dataset.add_ellipse_collection(10, label=-1)
dataset.change_collection_location(3, 0, -2)


df = pd.DataFrame(dataset.build_dataframe(), columns=["x", "y", "label"])
# print(df.head())

df["label"] = df["label"].apply(lambda x: 0 if x == -1 else x)

x_train = np.array(df.drop(columns="label"))
y_train = np.array(df["label"])

df["label"] = df["label"].astype(str)


class MultiLayerPerceptron():
    """
    A multi layer perceptron model.
    This one is designed to save the weight history at given intervals during the training process 
    for the purpose of giving the option to predict based on previous weights. 
    """
    def __init__(self, learning_rate: float, epochs: int, hidden_layers: list[int]) -> None:
        self.weights_history = []
        self.biases_history = []
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.hidden_layers = hidden_layers

    def get_history_length(self) -> int:
        return len(self.weights_history)

    def relu(self, x: float) -> float:
        """
        Rectified Linear Unit.
        Used as activation function in the hidden layers. 

        Args:
            x (float): value to pass through the function. 

        Returns:
            float: f(x)
        """
        return max(0, x)
    
    def relu_derivative(self, x: float) -> float:
        """
        Derivative of ReLU for backward propagation. 

        Args:
            x (float): value to pass through the function. 

        Returns:
            float: f"(x)
        """
        if x > 0:
            return 1
        else:
            return 0

    def sigmoid(self, x: float) -> float:
        """
        Sigmoid function used as activation in the final perceptron layer. 

        Args:
            x (float): value to pass through the function.

        Returns:
            float: f(x)
        """
        return 1/(1+np.exp(-x))

    def sigmoid_derivative(self, x: float) -> float:
        """
        Derivative of ReLU for backward propagation. 

        Args:
            x (float): value to pass through the function.

        Returns:
            float: f"(x)
        """
        return self.sigmoid(x)*(1 - self.sigmoid(x))
    
    def step_function(self, x: float) -> int:
        """
        step function with a threshold at 0.5.
        Used for the final classification by the perceptron. 

        Args:
            x (float): value to pass through the function.

        Returns:
            int: f(x)
        """
        if x > 0.5:
            return 1
        else:
            return 0

    def fit(self, x_train: np.array, y_train: np.array) -> list[float]:
        """
        Method for training the model. 

        Args:
            x_train (np.array): array of input vectors.
            y_train (np.array): array of labels.

        Returns:
            list[float]: mean squared error per epoch. 
        """
        x_dim = x_train.shape[1]
        out_dim = 1 # hard coded single perceptron as the final layer
        layers = self.hidden_layers
        layers.insert(0, x_dim)
        layers.append(out_dim)
        weights = []
        biases = []

        # init weight matrices
        for index, layer in enumerate(layers):
            try:
                weights.append(np.random.random((layers[index+1], layer)))
            except IndexError:
                pass # the last layer does not have weights after it so we pass
        # init biases
        for weight in weights:
            biases.append(np.random.random((len(weight),1)))

        # add initial matrices to history
        self.weights_history.append(weights.copy())
        self.biases_history.append(biases.copy())

        mean_squared_error = []
        for epoch in range(self.epochs):
            squared_error = []
            # train weights
            for y_index, x in enumerate(x_train):

                # feed forward
                # first output uses the input x
                outputs = []
                outputs.append(weights[0] @ x.reshape(1, len(x)).T + biases[0])
                # hidden layers uses the output of the previous layer
                for index, weight in enumerate(weights[1:]):
                    previous_activated = np.vectorize(self.relu)(outputs[index])
                    outputs.append(weight @ previous_activated + biases[index + 1])
                # final output is a regular perceptron and uses a different activation function
                last_output_sigmoid = np.vectorize(self.sigmoid)(outputs[-1])
                y_predicted = last_output_sigmoid[0][0]

                # backwards propagation for finding the gradients
                deltas = []
                # delta at the last position is the perceptron and uses a different activation function
                delta = (last_output_sigmoid - y_train[y_index]) * self.sigmoid_derivative(last_output_sigmoid)
                deltas.append(delta)
                for index in np.arange(-1,-(len(weights)),-1):
                    delta = (deltas[(-index-1)] @ weights[index]) * (np.vectorize(self.relu_derivative)(outputs[index-1])).T
                    deltas.append(delta)
                
                # gradient descent to update the weights
                for index, delta in enumerate(deltas):
                    reverse_index = -index-1
                    try:
                        weights[reverse_index] = weights[reverse_index] - self.learning_rate*np.kron(delta.T,outputs[reverse_index-1].T)
                        biases[reverse_index] = biases[reverse_index] - self.learning_rate*delta.T
                    except IndexError: # the final update uses the original output from x
                        weights[reverse_index] = weights[reverse_index] - self.learning_rate*np.kron(delta.T,x.reshape(1, len(x)))
                        biases[reverse_index] = biases[reverse_index] - self.learning_rate*delta.T
                
                # squared error
                squared_error.append((y_predicted-y_train[y_index])**2)
            mean_squared_error.append(float(sum(squared_error)/len(squared_error)))
            # add trained weights to history
            if epoch % 20 == 0:
                self.weights_history.append(weights.copy())
                self.biases_history.append(biases.copy())
        
        # add final model to history
        self.weights_history.append(weights.copy())
        self.biases_history.append(biases.copy())
        
        return mean_squared_error
    
    def predict(self, x: np.array, index: int = -1) -> list[int]:
        weights = self.weights_history[index]
        biases = self.biases_history[index]

        try:
            output = weights[0] @ x.T + biases[0]
        except ValueError: # if we only want prediction of a single point. broken for some reason
            output = weights[0] @ x.reshape(1, len(x)).T + biases[0]
        
        for index, weight in enumerate(weights[1:]):
            output = np.vectorize(self.relu)(output)
            output = weight @ output + biases[index + 1]

        output = np.vectorize(self.sigmoid)(output)
        output = np.vectorize(self.step_function)(output)
        return output[0]

def create_figure_mlp(model: MultiLayerPerceptron, features: np.array, labels: np.array, resolution: int=100, index: int=-1) -> go.Figure:
    """
    Plot the decision boundary of a mlp neural network.

    Args:
        model (MultiLayerPerceptron): Trained mlp model.
        features (np.array): feature of shape (n, 2).
        labels (np.array): labels of shape (n,).
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

    # Add contour plot
    fig.add_trace(go.Contour(
        x=np.linspace(x_min, x_max, resolution),
        y=np.linspace(y_min, y_max, resolution),
        z=pred,
        colorscale="RdBu",
        opacity=0.3,
        showscale=False,
        contours=dict(
            start=0,
            end=1,
            size=0.5
        )
    ))

    return fig

mlp = MultiLayerPerceptron(learning_rate=0.002, epochs=1000, hidden_layers=[10])

mse = mlp.fit(x_train, y_train)

pred = mlp.predict(x_train)

print(np.sum(np.abs(y_train-pred)))

fig = go.Figure()

# dummy trace. Added because of a bug in plotly frames which removes the first trace.
fig.add_trace(go.Scatter(
    x=df["x"][:1],
    y=df["y"][:1],
    mode="markers",
    marker={"color": "blue"},
    name="Class none"
    ))

# add traces for the classes in our dataframe
for class_value in ["0", "1"]:
    mask = df["label"] == class_value
    fig.add_trace(go.Scatter(
        x=df[mask]["x"],
        y=df[mask]["y"],
        mode="markers",
        marker=dict(color="red" if class_value == "0" else "blue"),
        name=f"Class {class_value}"
    ))


# fig.add_trace(go.Scatter(
#     x=df["x"],
#     y=df["y"],
#     mode="markers"
# ))

# Calculate the axis ranges
x_min, x_max = x_train[:, 0].min() - 0.5, x_train[:, 0].max() + 0.5
y_min, y_max = x_train[:, 1].min() - 0.5, x_train[:, 1].max() + 0.5

# Update the layout
fig.update_layout(
    title="Multi layer perceptron training process",
    xaxis_title="X",
    yaxis_title="Y",
    width=800,
    height=600,
    xaxis={"range": [x_min, x_max], "visible": False,
           "showticklabels": False, "hoverformat": ".2f"},
    yaxis={"range": [y_min, y_max], "visible": False,
           "showticklabels": False, "hoverformat": ".2f"},
)

fig.update_yaxes(
    scaleanchor="x",
    scaleratio=1,
)


length = mlp.get_history_length()

# Create and add frames
frames = [go.Frame(data=create_figure_mlp(model=mlp, features=x_train, labels=y_train, index=i).data, name=str(i))
          for i in range(length)]
fig.frames = frames

# Add slider and play button
fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(label="Play",
                     method="animate",
                     args=[None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}]),
                dict(label="Pause",
                     method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}])
            ]
        )
    ],
    sliders=[
        dict(
            steps=[
                dict(
                    method="animate",
                    args=[[str(i)], {"frame": {"duration": 100,
                                               "redraw": True}, "mode": "immediate"}],
                    label=str(i)
                )
                for i in range(length)
            ],
            transition={"duration": 0},
            x=0,
            y=0,
            currentvalue={"font": {"size": 12}, "prefix": "Iteration: ",
                          "visible": True, "xanchor": "center"},
            len=0.9,
        )
    ]
)


fig.show()
