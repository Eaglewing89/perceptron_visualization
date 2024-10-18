"""
Collection of necessary classes
"""

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
