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

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose

from qencore.nn import PlainICNN


@pytest.fixture
def icnn() -> PlainICNN:
    """
    Fixture to create an instance of MyClass with required arguments.

    Returns
    -------
    PlainICNN
        Test ICNN
    """
    x_list = [-1, -0.9, -0.8, -0.4, 0, 0.4, 0.8, 0.9, 1]
    y_list = [1.8, 0.6734, 0.3098, 0.0705, 0, 0.0705, 0.3098, 0.6734, 1.8]

    icnn1d = PlainICNN(input_size=1)
    icnn1d.train(x_list, y_list, lr=0.01, epochs=30_000, plot=False)

    return icnn1d


@pytest.mark.parametrize("tol", [1e-2, 1e-3])
def test_icnn_convexity(icnn: PlainICNN, tol: float) -> None:
    """
    Test the convexity property of the ICNN (Input Convex Neural Network).

    The test checks if the convexity property holds for the given ICNN instance.
    It generates random pairs of input vectors and verifies that the convexity
    inequality is satisfied within the given tolerance.

    Parameters
    ----------

    icnn : PlainICNN
        Instance of the ICNN class to be tested.
    tol : float
        Tolerance level for the allclose assertion.
    """

    data_dim = icnn.linear_weights[0].in_features

    zeros = np.zeros(100)
    for _ in range(100):
        x = torch.rand((100, data_dim))
        y = torch.rand((100, data_dim))

        fx = icnn.evaluate(x)
        fy = icnn.evaluate(y)

        for t in np.linspace(0, 1, 10):
            fxy = icnn.evaluate(t * x + (1 - t) * y)
            res = (t * fx + (1 - t) * fy) - fxy
            assert_allclose(np.minimum(res, 0), zeros, atol=tol)
