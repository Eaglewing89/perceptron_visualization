import numpy as np
import pandas as pd
from classes import PointCollection
import plotly_express as px
import plotly.graph_objects as go
import copy


dataset = PointCollection()
dataset.add_ellipse_collection(40, label=1)
dataset.change_collection_location(0, -1.5, 0)
dataset.add_ellipse_collection(40, label=1)
dataset.change_collection_location(1, 1.5, 0)
dataset.add_ellipse_collection(40, label=-1)
dataset.change_collection_location(2, 0, 1.5)
dataset.add_ellipse_collection(40, label=-1)
dataset.change_collection_location(3, 0, -1.5)
dataset.add_ellipse_collection(40, label=-1)
dataset.change_collection_location(4, 2, -2.5)


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

    def __init__(self, learning_rate: float, epochs: int, hidden_layers: list[int], save_interval: int = 100) -> None:
        """
        Init method

        Args:
            learning_rate (float): eta - how strong are each update. Recommend around 0.01 or less.
            epochs (int): how many times to loop through all points during training. 
            hidden_layers (list[int]): structure of hidden layers. Each element is nr of nodes. 
            save_interval (int): after how many epoch we store the weights. Defaults to 100.

        Raises:
            ValueError: raised if you do not follow type hints. 
        """
        self.weights_history = []
        self.biases_history = []
        try:
            self.save_interval = int(save_interval)
            self.learning_rate = float(learning_rate)
            self.epochs = int(epochs)
            for layer in hidden_layers:
                layer = int(layer)
            self.hidden_layers = list(hidden_layers)
        except ValueError as err:
            raise ValueError from err

    def get_history_length(self) -> int:
        """
        Get the number of weights stored in history.

        Returns:
            int: number of weights total.
        """
        return len(self.weights_history)

    def get_save_interval(self) -> int:
        """
        Get the number for save intervals.

        Returns:
            int: save interval number.
        """
        return self.save_interval

    def _relu(self, x: np.array) -> np.array:
        """
        Rectified Linear Unit.
        Used as activation function in the hidden layers. 

        Args:
            x (np.array): Array of values to pass through the function. 

        Returns:
            np.array: f(x) applied element-wise.
        """
        return np.maximum(0, x)

    def _relu_derivative(self, x: np.array) -> np.array:
        """
        Derivative of ReLU for backward propagation. 

        Args:
            x (np.array): Array of values to pass through the function. 

        Returns:
            np.array: f'(x) applied element-wise.
        """
        return np.where(x > 0, 1, 0)

    def _sigmoid(self, x: np.array) -> np.array:
        """
        Sigmoid function used as activation in the final perceptron layer. 

        Args:
            x (np.array): Array of values to pass through the function.

        Returns:
            np.array: f(x) applied element-wise.
        """
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, x: np.array) -> np.array:
        """
        Derivative of sigmoid for backward propagation. 

        Args:
            x (np.array): Array of values to pass through the function.

        Returns:
            np.array: f'(x) applied element-wise.
        """
        sigmoid_x = self._sigmoid(x)
        return sigmoid_x * (1 - sigmoid_x)

    def _step_function(self, x: np.array) -> np.array:
        """
        Step function with a threshold at 0.5.
        Used for final classification by the perceptron in batched predictions.

        Args:
            x (np.array): Array of values to pass through the function.

        Returns:
            np.array: Array with 1s where f(x) > 0.5 and 0s otherwise.
        """
        return np.where(x > 0.5, 1, 0)

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
        out_dim = 1  # hard coded single perceptron as the final layer
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
                pass  # the last layer does not have weights after it so we pass
        # init biases
        for weight in weights:
            biases.append(np.random.random((len(weight), 1)))

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
                    previous_activated = self._relu(outputs[index])
                    outputs.append(
                        weight @ previous_activated + biases[index + 1])
                # final output is a regular perceptron and uses a different activation function
                last_output_sigmoid = self._sigmoid(outputs[-1])
                y_predicted = last_output_sigmoid[0][0]

                # backwards propagation for finding the gradients
                deltas = []
                # delta at the last position is the perceptron and uses a different activation function
                delta = (last_output_sigmoid -
                         y_train[y_index]) * self._sigmoid_derivative(last_output_sigmoid)
                deltas.append(delta)
                for index in np.arange(-1, -(len(weights)), -1):
                    delta = (deltas[(-index-1)] @ weights[index]) * \
                        (self._relu_derivative(outputs[index-1])).T
                    deltas.append(delta)

                # gradient descent to update the weights
                for index, delta in enumerate(deltas):
                    reverse_index = -index-1
                    try:
                        weights[reverse_index] = weights[reverse_index] - \
                            self.learning_rate * \
                            np.kron(delta.T, outputs[reverse_index-1].T)
                        biases[reverse_index] = biases[reverse_index] - \
                            self.learning_rate*delta.T
                    except IndexError:  # the final update uses the original output from x
                        weights[reverse_index] = weights[reverse_index] - \
                            self.learning_rate * \
                            np.kron(delta.T, x.reshape(1, len(x)))
                        biases[reverse_index] = biases[reverse_index] - \
                            self.learning_rate*delta.T

                # squared error
                squared_error.append((y_predicted-y_train[y_index])**2)
            mean_squared_error.append(
                float(sum(squared_error)/len(squared_error)))
            # add trained weights to history
            if epoch % self.save_interval == 0:
                self.weights_history.append(weights.copy())
                self.biases_history.append(biases.copy())

        # add final model to history
        self.weights_history.append(weights.copy())
        self.biases_history.append(biases.copy())

        return mean_squared_error
    
    def fit2(self, x_train: np.array, y_train: np.array) -> list[float]:
        """
        Method for training the model using mini-batch gradient descent.

        Args:
            x_train (np.array): array of input vectors.
            y_train (np.array): array of labels.

        Returns:
            list[float]: mean squared error per epoch.
        """
        x_dim = x_train.shape[1]
        out_dim = 1  # hard coded single perceptron as the final layer
        batch_size = 32  # you can adjust this hyperparameter
        
        # Configure network architecture
        layers = self.hidden_layers
        layers.insert(0, x_dim)
        layers.append(out_dim)
        
        # Initialize weights and biases using He initialization
        weights = []
        biases = []
        for index, layer in enumerate(layers[:-1]):
            scale = np.sqrt(2.0 / layer)
            weights.append(np.random.randn(layers[index + 1], layer) * scale)
            biases.append(np.random.randn(layers[index + 1], 1) * scale)

        # Store initial state
        self.weights_history.append(copy.deepcopy(weights))
        self.biases_history.append(copy.deepcopy(biases))

        mean_squared_error = []
        n_samples = x_train.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))

        for epoch in range(self.epochs):
            # Shuffle data at the start of each epoch
            shuffle_idx = np.random.permutation(n_samples)
            x_shuffled = x_train[shuffle_idx]
            y_shuffled = y_train[shuffle_idx]
            
            epoch_errors = []

            for batch in range(n_batches):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, n_samples)
                
                # Get batch data
                x_batch = x_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                current_batch_size = end_idx - start_idx

                # Forward pass
                activations = [x_batch.T]  # Store all activations for backprop
                outputs = []  # Store all outputs before activation

                # Hidden layers
                for i in range(len(weights) - 1):
                    output = weights[i] @ activations[-1] + biases[i]
                    outputs.append(output)
                    activation = self._relu(output)
                    activations.append(activation)

                # Output layer (sigmoid activation)
                final_output = weights[-1] @ activations[-1] + biases[-1]
                outputs.append(final_output)
                y_pred = self._sigmoid(final_output)
                activations.append(y_pred)

                # Backward pass
                deltas = []
                
                # Output layer error
                delta = (y_pred - y_batch.reshape(1, -1)) * self._sigmoid_derivative(outputs[-1])
                deltas.append(delta)

                # Hidden layer errors
                for i in range(len(weights) - 2, -1, -1):
                    delta = (weights[i + 1].T @ deltas[0]) * self._relu_derivative(outputs[i])
                    deltas.insert(0, delta)

                # Update weights and biases
                for i, weight in enumerate(weights):
                    # Calculate average gradients over the batch
                    weight_gradient = deltas[i] @ activations[i].T / current_batch_size
                    bias_gradient = np.mean(deltas[i], axis=1, keepdims=True)
                    
                    # Update with momentum (optional improvement)
                    weight -= self.learning_rate * weight_gradient
                    biases[i] -= self.learning_rate * bias_gradient

                # Calculate batch error
                batch_error = np.mean((y_pred.T - y_batch) ** 2)
                epoch_errors.append(batch_error)

            # Store epoch metrics
            mean_squared_error.append(float(np.mean(epoch_errors)))
            
            # Save weights periodically
            if epoch % self.save_interval == 0:
                self.weights_history.append(copy.deepcopy(weights))
                self.biases_history.append(copy.deepcopy(biases))

        # Save final weights
        self.weights_history.append(copy.deepcopy(weights))
        self.biases_history.append(copy.deepcopy(biases))

        return mean_squared_error

    def predict(self, x: np.array, index: int = -1) -> list[int]:
        """
        Predict method. 

        Args:
            x (np.array): input data with shape (n, 2).
            index (int, optional): Which set of weights to use from history. Defaults to -1.

        Returns:
            list[int]: a list of predictions.
        """
        weights = self.weights_history[index]
        biases = self.biases_history[index]
        # feed forward
        try:
            output = weights[0] @ x.T + biases[0]
        except ValueError:  # if we only want prediction of a single point. broken for some reason
            output = weights[0] @ x.reshape(1, len(x)).T + biases[0]
        for index, weight in enumerate(weights[1:]):
            output = self._relu(output)
            output = weight @ output + biases[index + 1]
        output = self._sigmoid(output)
        # take the result through a step function to get exactly 0 or 1
        output = self._step_function(output)
        return output[0]

def create_figure_mlp(model: MultiLayerPerceptron, features: np.array, resolution: int=100, index: int=-1) -> go.Figure:
    """
    Plot the decision boundary of a mlp neural network. 2d inputs.

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

    trace = go.Contour(
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
    )

    return fig

epochs = 2000

mlp = MultiLayerPerceptron(learning_rate=0.005, epochs=epochs, hidden_layers=[20,2])

mse = mlp.fit2(x_train, y_train)

pred = mlp.predict(x_train)

print(np.sum(np.abs(y_train-pred)))

print(mlp.get_history_length())

def basefig():

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
    return fig

fig = basefig()


length = mlp.get_history_length()
iterations = [i*100 for i in range(length-1)]
iterations.append(epochs+1)

# Create and add frames
frames = [go.Frame(data=create_figure_mlp(model=mlp, features=x_train, index=i).data, name=str(i))
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
                    label=str(iterations[i])
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

