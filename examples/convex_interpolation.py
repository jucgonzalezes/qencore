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

import matplotlib.pyplot as plt
import numpy as np
import torch
import typer

from qencore.nn import MLP, PlainICNN
from qencore.splines import MultisegmentConvexInterpolation1D


def main(
    sigma: float = typer.Option(0.1, help="Sigma value"),
    gamma: float = typer.Option(1, help="Gamma value"),
):
    def f(x):
        return 0.4 * x**2 + 0.5 * x**4 - 1.6 * x**6 + 2.5 * x**10

    # Real Function
    x_new = list(np.linspace(-1, 1, 60))
    fx = [f(x_) for x_ in x_new]

    # Sampled data
    x = [-1, -0.9, -0.8, -0.4, 0, 0.4, 0.8, 0.9, 1]
    y = [1.8, 0.6734, 0.3098, 0.0705, 0, 0.0705, 0.3098, 0.6734, 1.8]

    # Splines
    # sigma, gamma = 0.1, 1
    msc_1d = MultisegmentConvexInterpolation1D(x, y, gamma, sigma)
    msc_1d.fit()
    y_new_splines = msc_1d.evaluate(x_new)
    print("1st part: Multisegment Convex Interpolation\n \t  Code sucessfully Executed")

    # MLP
    mlp_1d = MLP(input_size=1, neurons=[16, 16], output_size=1)
    mlp_1d.train(torch.Tensor(x), torch.Tensor(y), lr=0.01, epochs=30_000, plot=True)
    y_new_mlp = mlp_1d.evaluate(torch.Tensor(x_new))
    print("2nd part: Multilayer perceptron\n \t  Code sucessfully Executed")

    # ICNN
    icnn_1d = PlainICNN(input_size=1)
    icnn_1d.train(x, y, lr=0.01, epochs=30_000, plot=True)
    y_new_icnn = icnn_1d.evaluate(torch.Tensor(x_new))
    print("3rd part: ICNN\n \t  Code sucessfully Executed")

    # Plotting
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica"]

    plt.plot(x_new, y_new_icnn, lw=3, zorder=1, label="ICNN")
    plt.plot(x_new, y_new_mlp, lw=3, zorder=3, label="FNN")
    plt.plot(x_new, y_new_splines, lw=3, zorder=2, label="Convex Splines")

    plt.plot(x_new, fx, "k--", lw=1, zorder=4, label="Real function")
    plt.scatter(x, y, marker="o", color="k", label="Training points", s=30, zorder=5)

    plt.xlabel(r"$x$", size=12)
    plt.ylabel(r"$f(x)$", size=12)

    plt.legend()

    plt.show()


if __name__ == "__main__":
    typer.run(main)
