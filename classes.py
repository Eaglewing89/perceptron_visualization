"""
Collection of necessary classes
"""
from math import pi, sin, cos
from random import uniform, choice
import pandas as pd
import numpy as np
import graphviz


class Perceptron:
    """
    A simple implementation of the perceptron model.
    """

    def __init__(self, learning_rate=0.01) -> None:
        self.learning_rate = learning_rate
        self.weights = None

    def sign_function(self, weighted_sum: float) -> int:
        """
        A method for returning 1 for positive values and -1 for negative values

        Args:
            weighted_sum (float): The result of <w,x>

        Returns:
            int: 1 for positive values, -1 for negative values
        """
        if weighted_sum >= 0:
            return 1
        else:
            return -1

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series, epochs: int, learning_rate: float = 0.01) -> list[np.array]:
        """
        Method for training the perceptron weights based on the training data

        Args:
            x_train (pd.DataFrame): training data
            y_train (pd.Series): training labels
            epochs (int): how many times we should go through all training data with the algorithm
            learning_rate (float, optional): the rate or how much we should update the weights during the training process. Values above 1 are not recommended. Defaults to 0.01.

        Returns:
            list[np.array]: a list containing a history of weights for plotting purposes
        """
        length, dimension = x_train.shape
        self.weights = np.random.rand(dimension + 1)
        # self.weights = np.array([0,1.1,-1])
        w_history = []
        for _ in range(epochs):
            for index in range(length):
                weighted_sum = (
                    x_train[index]@self.weights[1:]) + self.weights[0]
                prediction = self.sign_function(weighted_sum)
                loss = (prediction-y_train[index])/2
                self.weights[1:] -= learning_rate * loss * x_train[index]
                self.weights[0] -= learning_rate * loss

            w_history.append(self.weights.copy())

        return w_history


class PointCollection:
    def __init__(self) -> None:
        self.collections_points = []
        self.collections_locations = []
        self.collections_rotations = []
        self.collections_labels = []
        self.collections_scales = []

    def _create_ellipse_points(self, number: int, a: float = 1, b: float = 1, max_angle: float = 2*pi) -> np.array:
        """
        randomises points in the shape of an ellipse. uses skewed_uniform to ensure that most points are near the circumference to make it looks nicer

        Args:
            number (int): number of desired points
            label (int): the class of all points to make it simpler to create a dataframe
            a (float, optional): ellipse constant. Defaults to 1.
            b (float, optional): ellipse constant. Defaults to 1.

        Returns:
            list[float]: list of points with class label
        """
        points = np.array([]).reshape(0, 2)
        for _ in range(number):
            angle = uniform(0, max_angle)
            new_point = np.array(
                [[self._skewed_uniform(a*cos(angle)), self._skewed_uniform(b*sin(angle))]])
            points = np.concat((points, new_point))
        return points

    def _create_rectangle_points(self, number: int, short_side: float) -> np.array:
        random_choice = [-1, 1]
        points = np.array([]).reshape(0, 2)
        for _ in range(number):
            if choice(random_choice) == 1:
                x = 1*short_side
                y = uniform(0, 1)
            else:
                x = uniform(0, 1)*short_side
                y = 1
            skew = self._skewed_uniform(1)
            new_point = np.array(
                [[x*choice(random_choice)*skew, y*choice(random_choice)*skew]])
            points = np.concat((points, new_point))
        return points

    def _move_points(self, points: np.array, location: list[float]) -> np.array:
        """
        move a set of points in 2d.

        Args:
            points (np.array): n by 2 matrix.
            location (list[float]): A list with [x_location, y_location]

        Returns:
            np.array: 2 by n matrix at the new location
        """
        added_distance = np.array([location])
        return points+added_distance

    def _rotate_points(self, points: np.array, angle: float) -> np.array:
        """
        rotate a set of points around origin

        Args:
            points (np.array): which set of points to rotate
            angle (float): angle of rotation

        Returns:
            np.array: rotated set of points
        """
        rotation = np.array(
            [[cos(angle), sin(angle)], [-sin(angle), cos(angle)]])
        return points @ rotation

    def _scale_points(self, points: np.array, scale: float) -> np.array:
        scaler = np.array([[scale, 0], [0, scale]])
        return np.dot(points, scaler)

    def _skewed_uniform(self, value: float) -> float:
        """
        skews a uniform(0,1) towards 1, multiplies this with a value and returns it

        Args:
            value (float): just a float

        Returns:
            float: skewed float
        """
        uni = uniform(0, 1)
        uni = np.sqrt(uni*2)
        if uni > 1:
            uni = 1
        return uni*value

    def add_ellipse_collection(self, number_of_points: int, a: float = 1, b: float = 1, label: int = 1, rotation_angle: float = 0, position_x: float = 0, position_y: float = 0, max_ellipse_angle: float = 2*pi, scale: float = 1) -> None:
        """
        User method that adds an ellipse shape and all necessary variables.

        Args:
            number_of_points (int): How many points the ellipse consists of.
            a (float): excentricity a.
            b (float): excentricity b.
            label (int): class label of the collection.
            rotation_angle (float, optional): 0-2pi angle. Defaults to 0.
            position_x (float, optional): position on x axis. Defaults to 0.
            position_y (float, optional): position on y axis. Defaults to 0.

        Raises:
            TypeError: Raised if number_of_points is not int, or if rotation_angle, position_x, position_y is neither int nor float.
            ValueError: Raised if excentricity values are 0 or less.
            IndexError: Raised if the class is not -1 or 1. 
        """
        if not isinstance(number_of_points, int):
            raise TypeError
        if a <= 0 or b <= 0:
            raise ValueError
        if not label in [-1, 1]:
            raise IndexError
        if not isinstance(rotation_angle, (int, float)):
            raise TypeError
        if not isinstance(position_x, (int, float)):
            raise TypeError
        if not isinstance(position_y, (int, float)):
            raise TypeError
        if not isinstance(scale, (int, float)):
            raise TypeError
        self.collections_points.append(
            self._create_ellipse_points(number=number_of_points, a=a, b=b, max_angle=max_ellipse_angle))
        self.collections_locations.append(
            [float(position_x), float(position_y)])
        self.collections_rotations.append(float(rotation_angle))
        self.collections_labels.append(label)
        self.collections_scales.append(float(scale))

    def add_rectangle_collection(self, number_of_points: int, short_side: float = 1) -> None:
        try:
            self.collections_points.append(
                self._create_rectangle_points(number_of_points, short_side))
            self.collections_locations.append([float(0), float(0)])
            self.collections_rotations.append(float(0))
            self.collections_labels.append(int(1))
            self.collections_scales.append(float(1))
        except ValueError as exc:
            raise ValueError from exc

    def change_collection_rotation(self, index: int, rotation_angle: float) -> None:
        """
        Changes the listed rotation angle of a collection.

        Args:
            index (int): Collection index.
            rotation_angle (float): New rotation value.

        Raises:
            TypeError: Raised if rotation_angle is neither int nor float.
            IndexError: Raised if index does not exist.
        """
        if not isinstance(rotation_angle, (int, float)):
            raise TypeError
        if not index in range(len(self.collections_rotations)):
            raise IndexError
        self.collections_rotations[index] = rotation_angle

    def change_collection_location(self, index: int, position_x: float, position_y: float) -> None:
        """
        Changes the listed location of a collection.

        Args:
            index (int): Collection index.
            position_x (float): New x-axis position.
            position_y (float): New y-axis position.

        Raises:
            TypeError: Raised if position_x or position_y is neither int nor float.
            IndexError: Raised if index does not exist.
        """
        if not isinstance(position_x, (int, float)):
            raise TypeError
        if not isinstance(position_y, (int, float)):
            raise TypeError
        if not index in range(len(self.collections_rotations)):
            raise IndexError
        self.collections_locations[index] = [
            float(position_x), float(position_y)]

    def change_collection_scale(self, index: int, scale: float) -> None:
        """
        Saves a new scale for a collection of points in self.collections_scales list.

        Args:
            index (int): index of the point collection.
            scale (float): the new scale of the point collection.

        Raises:
            IndexError: Raised if the index was not found.
            ValueError: Raised if the scale value cannot be converted to a float.
        """
        try:
            self.collections_scales[index] = float(scale)
        except IndexError as exc:
            raise IndexError from exc
        except ValueError as exc:
            raise ValueError from exc

    def change_collection_label(self, index: int, label: int):
        """
        Change the label of a collection.

        Args:
            index (int): Collection index.
            label (int): New label.

        Raises:
            ValueError: Raised if label is not int or not -1 or 1.
            IndexError: Raised if collection index not found.
        """
        try:
            if label in [-1, 1]:
                self.collections_labels[index] = int(label)
            else:
                raise ValueError
        except IndexError as exc:
            raise IndexError from exc

    def remove_collection(self, index: int) -> None:
        """
        Removes stored information about a collection of points from all instance variables.

        Args:
            index (int): Index of the collection of points to be removed.

        Raises:
            IndexError: Raised if index could not be found. 
        """
        try:
            self.collections_labels.pop(index)
            self.collections_locations.pop(index)
            self.collections_points.pop(index)
            self.collections_rotations.pop(index)
            self.collections_scales.pop(index)
        except IndexError as exc:
            raise IndexError from exc

    def number_of_collections(self) -> int:
        """
        Checks and returns the total number of collections.

        Returns:
            int: Number of collections stored.
        """
        return len(self.collections_labels)

    def get_collection_location(self, index: int):
        return self.collections_locations[index]

    def get_collection_scale(self, index: int):
        return self.collections_scales[index]

    def get_collection_rotation(self, index: int):
        return self.collections_rotations[index]

    def get_collection_label(self, index: int):
        return self.collections_labels[index]

    def build_single_collection_dataframe(self, index: int):
        try:
            points = self.collections_points[index]
            points = self._scale_points(points, self.collections_scales[index])
            points = self._rotate_points(
                points, self.collections_rotations[index])
            points = self._move_points(
                points, self.collections_locations[index])
            label = self.collections_labels[index]
            length = points.shape[0]
            label_array = (np.ones(length)*label).reshape(length, 1)
            points = np.concat((points, label_array), axis=1)
            df = pd.DataFrame(points, columns=["x", "y", "label"])
            df["label"] = df["label"].astype(int)
            return df
        except IndexError as exc:
            raise IndexError from exc

    def build_dataframe(self, total_x_location: float = None, total_y_location: float = None, total_rotation_angle: float = None) -> pd.DataFrame:
        """
        Uses the stored information to perform location and rotation changes to all collections.
        Adds their label to their respective arrays.
        Concatenates all collections.
        Adds this data to a dataframe and returns it.

        Returns:
            pd.DataFrame: Dataframe containing all information on stored collections. 
        """
        all_points = np.array([]).reshape(0, 2)
        all_labels = np.array([]).reshape(0, 1)
        for index, element in enumerate(self.collections_points):
            points = element
            points = self._scale_points(points, self.collections_scales[index])
            points = self._rotate_points(
                points, self.collections_rotations[index])
            points = self._move_points(
                points, self.collections_locations[index])
            label = self.collections_labels[index]
            length = points.shape[0]
            label_array = (np.ones(length)*label).reshape(length, 1)
            all_labels = np.concat((all_labels, label_array), axis=0)
            all_points = np.concat((all_points, points), axis=0)
        if total_rotation_angle:
            all_points = self._rotate_points(
                points=all_points, angle=total_rotation_angle)
        if total_x_location and total_y_location:
            all_points = self._move_points(points=all_points, location=[
                                           total_x_location, total_y_location])
        matrix = np.concat((all_points, all_labels), axis=1)
        df = pd.DataFrame(matrix, columns=["x", "y", "label"])
        df["label"] = df["label"].astype(int)
        return df


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


class NetworkGraph():
    """
    Class for creating a graph of a neural network. 
    """

    def __init__(self) -> None:
        self.input_nodes = 2
        self.output_nodes = 1
        self.max_visible_nodes = 5
        self.edge_color = "#888888"

    def get_graph(self, hidden_layers: list[int]) -> graphviz.Digraph:
        """
        Creates a graph of a neural network based on hidden layers. 

        Args:
            hidden_layers (list[int]): a list containing nodes per layer.

        Returns:
            graphviz.Digraph: graphviz object to be displayed. 
        """
        # Base graph
        graph = graphviz.Digraph()
        graph.attr(rankdir="LR",
                   bgcolor="#262730",
                   edge="black",
                   style="filled",
                   size="5,2.5",
                   ratio="fill")

        # Input layer
        with graph.subgraph(name="input") as c:
            c.attr(rank="same")
            input_labels = ["x", "y"]
            for i in range(self.input_nodes):
                c.node(f"i{i}", input_labels[i],
                       style="filled",
                       fillcolor="#90EE90",
                       shape="circle",
                       fixedsize="true",
                       width="0.4",
                       height="0.4")

        # Hidden layers
        for layer_idx, nodes in enumerate(hidden_layers):
            with graph.subgraph(name=f"hidden_{layer_idx}") as c:
                c.attr(rank="same")
                visible_nodes = min(nodes, self.max_visible_nodes)
                for node in range(visible_nodes):
                    c.node(f"h{layer_idx}_{node}", "",
                           style="filled",
                           fillcolor="#FAFAFA",
                           shape="circle",
                           fixedsize="true",
                           width="0.4",
                           height="0.4")

                if nodes > self.max_visible_nodes:
                    c.node(f"h{layer_idx}_more", self._get_node_label(nodes, self.max_visible_nodes),
                           style="filled",
                           fillcolor="#FAFAFA",
                           shape="note",
                           fontsize="10")

        # Output layer
        with graph.subgraph(name="output") as c:
            c.attr(rank="same")
            for i in range(self.output_nodes):
                c.node(f"o{i}", "Output",
                       style="filled",
                       fillcolor="#FFB6C1",
                       shape="circle",
                       fixedsize="true",
                       width="0.4",
                       height="0.4")

        # Connect input to first hidden layer
        first_hidden_visible = self._get_visible_node_count(hidden_layers[0])
        for i in range(self.input_nodes):
            for j in range(first_hidden_visible):
                graph.edge(f"i{i}", f"h0_{j}", dir="none",
                           color=self.edge_color)

        # Connect hidden layers
        for layer_idx in range(len(hidden_layers) - 1):
            current_visible = self._get_visible_node_count(
                hidden_layers[layer_idx])
            next_visible = self._get_visible_node_count(
                hidden_layers[layer_idx + 1])

            for node1 in range(current_visible):
                for node2 in range(next_visible):
                    graph.edge(f"h{layer_idx}_{node1}", f"h{layer_idx+1}_{node2}",
                               dir="none", color=self.edge_color)

        # Connect last hidden layer to output
        last_hidden_visible = self._get_visible_node_count(hidden_layers[-1])
        for i in range(last_hidden_visible):
            for j in range(self.output_nodes):
                graph.edge(f"h{len(hidden_layers)-1}_{i}", f"o{j}",
                           dir="none", color=self.edge_color)

        return graph

    def _get_node_label(self, total_nodes: int, visible_nodes: int) -> str:
        """
        Returns the node labels accounting for overflow nodes.

        Args:
            total_nodes (_type_): nodes in layer.
            visible_nodes (_type_): max nodes allowed.

        Returns:
            str: name of node.
        """
        if total_nodes > visible_nodes:
            return f"... ({total_nodes - visible_nodes} more)"
        return ""

    def _get_visible_node_count(self, total_nodes: int) -> int:
        """
        Return amount of visible nodes.

        Args:
            total_nodes (int): nodes in layer.

        Returns:
            int: amount visible nodes.
        """
        return min(total_nodes, self.max_visible_nodes)
