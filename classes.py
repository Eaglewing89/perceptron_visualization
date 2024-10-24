"""
Collection of necessary classes
"""
from math import pi, sin, cos
from random import uniform, choice
import pandas as pd
import numpy as np


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
        for epoch in range(epochs):
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
        random_choice = [-1,1]
        points = np.array([]).reshape(0, 2)
        for _ in range(number):
            if choice(random_choice) == 1:
                x = 1*short_side
                y = uniform(0,1)
            else:
                x = uniform(0,1)*short_side
                y = 1
            skew = self._skewed_uniform(1)
            new_point = np.array([[x*choice(random_choice)*skew,y*choice(random_choice)*skew]])
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

    def add_rectangle_collection(self, number_of_points: int, short_side: float) -> None:
        try:
            self.collections_points.append(self._create_rectangle_points(number_of_points, short_side))
            self.collections_locations.append([float(0),float(0)])
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
