# This code is part of qencore.
#
# (C) Copyright IBM 2024
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from typing import List

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch import autograd
from torch.nn.modules.loss import _Loss


class BaseNN(nn.Module):
    """
    Base class for classes MLP and PlainICNN. It inherits from `torch.nn.Module`.

    Methods
    -------
    evaluate(input_vector)
        Evaluates a model on a given dataset.
    """

    def __init__(self):
        super(BaseNN, self).__init__()

    def evaluate(self, input_vector: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the model using the provided dataset.

        Parameters
        ----------
        input_vector : torch.Tensor
            Dataset to evaluate.

        Returns
        -------
        torch.Tensor
            Model values for the dataset X.
        """
        output = torch.zeros(len(input_vector))

        with torch.no_grad():
            for inx, x in enumerate(input_vector):
                test_input = torch.tensor(
                    [[x]], dtype=torch.float32
                )  # torch.tensor([[0.3]], dtype=torch.float32)
                predicted_output = self(test_input).item()
                output[inx] = predicted_output

        return output


class MLP(BaseNN):
    """
    This class implements feedforward neural network with two hidden layers.
    It inherits from `BaseNN`.

    Parameters
    ----------
    input_size : int
        The number of input features.
    neurons: List[int]
        The number of neurons in each hidden layer.
    output_size: int
        The number of output features.

    Attributes
    ----------
    layer1 : torch.nn.Linear
        Linear transformation for the first hidden layer.
    activation1 : nn.ReLU
        ReLU activation function for the first hidden layer.
    layer2: torch.nn.Linear
        Linear transformation for the second hidden layer.
    activation2 : nn.ReLU
        ReLU activation for the second hidden layer.
    layer3 : nn.Linear
        Linear transformation for the output layer.

    Methods
    -------
    forward(x)
        Defines the forward pass of the network.
    train(nodes, y, lr=0.01, epochs=20_000, criterion=None, optimizer=None)
        Trains the model using the provided dataset, loss function, and optimizer.
    """

    def __init__(
        self, input_size: int = 1, neurons: List[int] = [16, 16], output_size: int = 1
    ) -> None:
        """
        Initializes the class MLP.

        Parameters
        ----------
        input_size : int
            The number of input features.
        neurons: List[int]
            The number of neurons in each hidden layer.
        output_size: int
            The number of output features.
        """
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, neurons[0])
        self.activation1 = nn.ReLU()
        self.layer2 = nn.Linear(neurons[0], neurons[1])
        self.activation2 = nn.ReLU()
        self.layer3 = nn.Linear(neurons[1], output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_size)

        Returns
        -------
        torch.Tensor
            Outpur tensor of shape (batch_size, output_size)
        """
        x = self.layer1(x)
        x = self.activation1(x)
        x = self.layer2(x)
        x = self.activation2(x)
        x = self.layer3(x)
        return x

    def train(  # type: ignore
        self,
        nodes: torch.Tensor,
        y: torch.Tensor,
        criterion: _Loss = None,  # type: ignore
        optimizer: optim.Optimizer = None,  # type: ignore
        lr: float = 0.01,
        epochs: int = 20_000,
        plot: bool = False,
    ) -> List[float]:
        """
        Trains the model using the provided dataset, loss function, optimizer, and
        training parameters.

        Parameters
        ----------
        nodes : torch.Tensor
            Training features
        y : torch.Tensor
            Training ouputs
        criterion : torch.nn.Module
            Loss function
        optimizer : torch.optim.Optimizer
            Optimizer
        lr : float
            Learning rate
        epochs : int
            Number of training epochs
        plot : bool, optional
            If true it plots the training error during training

        Returns
        -------
        List[float]
            Training losses

        """
        crit = criterion if criterion else nn.MSELoss()
        opt = optimizer if optimizer else optim.SGD(self.parameters(), lr=lr)

        nodes = torch.FloatTensor([_x for _x in nodes]).unsqueeze(1)
        y = torch.FloatTensor([_y for _y in y]).unsqueeze(1)
        train_losses = []

        for epoch in range(epochs):
            # Forward pass
            outputs = self(nodes)
            loss = crit(outputs, y)
            train_losses.append(loss.item())

            # Backward pass and optimization
            opt.zero_grad()
            loss.backward()
            opt.step()

            if (epoch + 1) % (epochs / 10) == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

        # Plot the training loss
        if plot:
            plt.plot(train_losses, label="Training Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.yscale("log")
            plt.legend()
            plt.show()

        return train_losses


class PositiveLinear(nn.Linear):
    """
    Positive Linear layer implemented with a softplus of factor beta.
    It inherits from `torch.nn.Linear`.

    Parameters
    ----------
    *args :
        args for torch.nn.Linear
    *kwargs :
        kwargs for torch.nn.Linear
    beta : int, optional
        Scaling factor for `torch.nn.functional.softplus` (default is 1)

    Atributes
    ---------
    beta : int
        Scaling factor for `torch.nn.functional.softplus`

    Methods
    -------
    forward(x)
        Defines the forward pass of the layer
    kernel()
        Defines the softplus kernel function
    """

    def __init__(self, *args, beta=1.0, **kwargs):
        """
        Initializes the class PositiveLinear.

        Parameters
        ----------
        *args :
            args for torch.nn.Linear
        *kwargs :
            kwargs for torch.nn.Linear
        beta : int, optional
            Scaling factor for `torch.nn.functional.softplus` (default is 1)
        """
        super(PositiveLinear, self).__init__(*args, **kwargs)
        self.beta = beta

    def forward(self, x):
        """
        Defines the forward pass of the network as a softplus transformation of
        the input.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_size)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_size)
        """
        return nn.functional.linear(x, self.kernel(), self.bias)

    def kernel(self):
        """
        Softplus kernel for the weights of the layer.

        Returns
        -------
        torch.Tensor
            Outpur tensor of shape (batch_size, output_size)
        """
        return nn.functional.softplus(self.weight, beta=self.beta)


class PlainICNN(BaseNN):
    """
    Implemented of the Input Convex Neural Network architecture introduced
    by Amos, Xu, and Kolter (2017).
    It inherits from `BaseNN`.

    Parameters
    ----------
    input_size : int
        Data input size
    units : List[int]
        Number of neurons per hidden layer

    Atributes
    ---------
    units : List[int]
        Number of neurons per hidden layer
    positive_weights : ...
        Set of positive layers
    linear_weights : ...
        Set of unconstrained layers

    Methods
    -------
    forward(x)
        Defines the forward pass of the layer
    clamp_w()
        Replaces negative weights in W and A by 0
    transport(x)
        Computes the local gradient of the NN on x
    train(nodes, y, lr=0.1, epochs=20_000, plot=False, criterion=None, optimizer=None)
        Trains the model using the provided dataset, loss function, and optimizer.
    """

    def __init__(
        self,
        input_size,
        units=[64, 64, 64, 64],
    ):
        """
        Initializes the class PlainICNN. It creates the architecture introduced
        by Amos, Xu, and Kolter in Input Convex Neural Network (2017). The
        network has two branches:

        - First branch (positive_weigths): A regular fully connected layer between the
          input layer and the first hidden layer followed by fully connected layers
          with positive weigths. The positive layers are implemented via the class
          PositiveLinear that uses a softplus to transform the weights. This branch is
          introduced to enforce the convexity of the output.

        - Second branch (linear_weights): A set of skip connections connecting the
          input layer directly with each hidden layer. This branch is introduces to
          guarantee expresibility of the NN.

        Parameters
        ----------
        input_size : int
            Data input size
        units : List[int]
            Number of neurons per hidden layer
        """
        super(PlainICNN, self).__init__()
        self.units = units + [1]  # Output-layer

        self.positive_weights = nn.ModuleList(
            [
                PositiveLinear(input_size, output_size, bias=False)
                for input_size, output_size in zip(self.units[:-1], self.units[1:])
            ]
        )
        self.linear_weights = nn.ModuleList(
            [
                nn.Linear(input_size, output_size, bias=True)
                for output_size in self.units[1:]
            ]
        )

    def forward(self, x):
        """
        Defines the forward pass of the network.

        To guarantee convexity the activation function must be convex. In
        this implementation we employ LeakyReLU. Additionally we square the
        outputs of the first layer to ensure positivity of the inputs of the
        subsequent layers.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_size)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_size)
        """
        # First Hidden layer
        z = nn.LeakyReLU(0.2)(self.linear_weights[0](x))
        z = z * z

        # Consecutive Hidden layers
        for positive_weight, linear_weight in zip(
            self.positive_weights[:-1], self.linear_weights[1:-1]
        ):
            z = nn.LeakyReLU(0.2)(positive_weight(z) + linear_weight(x))

        # Output Layer
        y = self.positive_weights[-1](z) + self.linear_weights[-1](x)
        return y

    def clamp_w(self):
        """
        Clamps the weigths of the NN.
        """
        for positive_weight in self.positive_weights:
            positive_weight.weight.data = positive_weight.weight.data.clamp(min=0)

    def transport(self, x):
        """
        Computes the gradient of the Neural Network evaluated on x.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_size)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_size)
        """
        assert x.requires_grad

        (output,) = autograd.grad(
            self.forward(x),
            x,
            create_graph=True,
            only_inputs=True,
            grad_outputs=torch.ones((x.size()[0], 1), device=x.device).float(),
        )
        return output

    def train(
        self,
        nodes,
        y,
        lr=0.1,
        epochs=20_000,
        criterion=None,
        optimizer=None,
        plot=False,
    ):
        """
        Trains the model using the provided dataset, loss function, optimizer, and
        training parameters.

        Parameters
        ----------
        nodes : torch.Tensor
            Training features
        y : torch.Tensor
            Training ouputs
        lr : float
            Learning rate
        epochs : int
            Number of training epochs
        criterion : torch.nn.Module
            Loss function
        optimizer : torch.optim.Optimizer
            Optimizer
        plot : bool, optional
            If true it plots the training error during training

        Returns
        -------
        List[float]
            Training losses

        """
        crit = criterion if criterion else nn.MSELoss()
        opt = optimizer if optimizer else optim.Adam(self.parameters(), lr=lr)

        nodes = torch.FloatTensor([_x for _x in nodes]).unsqueeze(1)
        y = torch.FloatTensor([_y for _y in y]).unsqueeze(1)
        train_losses = []

        for epoch in range(epochs):
            # Forward pass
            outputs = self(nodes)
            loss = crit(outputs, y)
            train_losses.append(loss.item())

            # Backward pass and optimization
            opt.zero_grad()
            self.clamp_w()
            loss.backward()
            opt.step()
            self.clamp_w()

            # Print training progress
            if (epoch + 1) % (epochs / 10) == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}")

        if plot:
            # Plot the training loss
            plt.plot(train_losses, label="Training Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.yscale("log")
            plt.legend()
            plt.show()

        return train_losses
