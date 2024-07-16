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

from math import ceil
from typing import Any, List

import numpy as np
from scipy import linalg


class MultisegmentConvexInterpolation1D:
    """
    Multisegment Convex Regression Class.

    This class performs multisegment convex interpolation on a given dataset.

    ...

    Attributes
    ----------
    nodes : list[float]
        The set of nodes i.e. the set of points in the x-axis that define the
        interpolation problem.
    y: list[float]
        The set of images i.e. the set of points in the y-axis that define the
        interpolation problem.
    gamma : float
        Defines the shape of the interpolating basis. The case gamma = 1 is equivalent
        to convex Hermite interpolation.
    sigma : float
        Value of the regularization strength of the empirical slopes. It should be a
        float value in the range [0.1, 0.2]

    Methods
    -------
    swam(...)
        ...
    compute_convex_orders(None)
        ...
    fit(...)
        ...
    evaluate(...)
        ...
    """

    def __init__(
        self, nodes: List[float], y: List[float], gamma: float = 1, sigma: float = 0.1
    ) -> None:
        """
        Parameters
        ----------
        nodes : list[float]
        The set of nodes i.e. the set of points in the x-axis that define the
        interpolation problem.
        y: list[float]
            The set of images i.e. the set of points in the y-axis that define the
            interpolation problem.
        gamma : float, optional
            Defines the shape of the interpolating basis. The case gamma = 1 is
            equivalent to convex Hermite interpolation (default is 1)
        sigma : float, optional
            Value of the regularization strength of the empirical slopes. It should be
            a float value in the range [0.1, 0.2] (default is 0.1)
        """
        self.nodes = nodes
        self.y = y
        self.gamma = gamma
        self.sigma = sigma
        self.fitted = False

    def swam(self) -> List[float]:
        """
        Slopes Weighted Averaging Method. It computes a weighted average of the empiral
        slopes defined by the input dataset.

        Returns
        -------
        List[float]
            List with the slopes estimated by the swam method for all the segments.

        Raises
        ------
        AssertionError
            If the dimensions of X and y do not coincide or if gamma lies outside
            [0.1, 0.2]
        """
        assert len(self.nodes) == len(self.y), "Vector sizes must be coincide."
        assert 0.1 <= self.sigma <= 0.2, r"sigma must be in [0.1, 0.2]"

        m = len(self.nodes)
        s = np.zeros(m)

        for inx in range(len(self.nodes)):
            if inx == 0:
                emp_s = (
                    (self.y[1] - self.y[0]) / (self.nodes[1] - self.nodes[0])
                    if (self.nodes[1] - self.nodes[0]) != 0
                    else 0
                )
                s[0] = (1 - self.sigma * np.sign(emp_s)) * emp_s

            elif inx == m - 1:
                emp_s = (
                    (self.y[m - 1] - self.y[m - 2])
                    / (self.nodes[m - 1] - self.nodes[m - 2])
                    if (self.nodes[m - 1] - self.nodes[m - 2]) != 0
                    else 0
                )
                s[m - 1] = (1 - self.sigma * np.sign(emp_s)) * emp_s

            else:
                emp_sp = (
                    (self.y[inx + 1] - self.y[inx])
                    / (self.nodes[inx + 1] - self.nodes[inx])
                    if (self.nodes[inx + 1] - self.nodes[inx]) != 0
                    else 0
                )
                emp_sm = (
                    (self.y[inx] - self.y[inx - 1])
                    / (self.nodes[inx] - self.nodes[inx - 1])
                    if (self.nodes[inx] - self.nodes[inx - 1]) != 0
                    else 0
                )

                delta_p = abs(self.nodes[inx + 1] - self.nodes[inx])
                delta_m = abs(self.nodes[inx] - self.nodes[inx - 1])

                s[inx] = (delta_m * emp_sm + delta_p * emp_sp) / (delta_p + delta_m)

        self.slopes = s

        return list(s)  # TODO: The returned value can be removed after a refactoring

    def compute_convex_orders(self) -> List[int]:
        """
        Computes the spline order in each segment to guarantee convexity of the
        interpolating function.

        Returns
        -------
        List[int]
            List of integers representing the splines order in each segment

        Raises
        ------
        AssertionError
            If the slopes list is not set or if it has a length different to that of
            the lists of nodes and images.
        """

        assert len(self.nodes) == len(self.slopes), "Vector sizes must coincide."

        def gamma_function(alpha: float) -> int:
            """
            Computes the smallest even number greater than or equal to alpha.

            Parameters
            ----------
            alpha : float
                Positive number to evaluate.

            Returns
            -------
            int
                The smallest even number greater than or equal to alpha.
            """
            return ceil(alpha) if ceil(alpha) % 2 == 0 else ceil(alpha) + 1

        def find_minimum_n_for_convexity(ns: int, nb: int) -> int:
            """
            Computes the smallest splines order to guarantee convexity.

            Parameters
            ----------
            ns : int
                Minimum value of n for the positive-curvature quadratic equation to
                vanish.
            nb : int
                Maximum value of n for the positive-curvature quadratic equation to
                vanish.

            Returns
            -------
            int
                The smallest even value greater than or equal to 4 of the splines order
                to guarantee convexity of the interpolating function.
            """
            return 4 if (ns >= 4 or nb <= 4) else gamma_function(nb)

        m = len(self.nodes)
        nc = np.zeros(m - 1)

        for inx in range(m - 1):
            sb = (self.y[inx + 1] - self.y[inx]) / (
                self.nodes[inx + 1] - self.nodes[inx]
            )
            (n1, n2) = (
                (self.slopes[inx + 1] - self.slopes[inx]) / (sb - self.slopes[inx]),
                (self.slopes[inx + 1] - self.slopes[inx]) / (self.slopes[inx + 1] - sb),
            )
            (ns, nb) = (min(n1, n2), max(n1, n2))
            nc[inx] = find_minimum_n_for_convexity(ns, nb)

        self.spline_orders = nc

        return list(nc)  # TODO: The returned value can be removed after a refactoring

    def fit(self) -> Any:
        """
        Computes the interpolation parameters to fit the dataset.

        The method first applies the swam algorithm to compute the local Slopes to be
        used by the interpolator at each node in X. Then, it computes the spline order
        in each segment using the nodes X, the images y, and the slopes s.

        For each segment the method solves the linear system A x = b, with A the matrix:

        [[a0, b0, 0, 1],

        [b0, a0, l, 1],

        [a1, -b1, 1, 0],

        [b1, -a1, 1, 0]],

        the vector x = [a, b, c, d] of unknown parameters, and the vector
        b = [y_i, y_{i+1}, s_i, s_{i+1}]. The values of a0, b0, a1, b1, and lambda are
        computed internally following the formulas detailed in the paper.
        """

        sp = self.swam()
        convex_orders = self.compute_convex_orders()

        number_of_segments = len(self.nodes) - 1
        regression_parameters = []

        for segment in range(number_of_segments):
            n = convex_orders[segment]
            lambda_var = self.nodes[segment + 1] - self.nodes[segment]

            # Bar constants
            a0b = (1 / (n * (n - 1))) * (1 / 2 * (1 - self.gamma)) ** n
            b0b = (1 / (n * (n - 1))) * (1 / 2 * (1 + self.gamma)) ** n
            a1b = (-1 / (n - 1)) * (1 / 2 * (1 - self.gamma)) ** (n - 1)
            b1b = (1 / (n - 1)) * (1 / 2 * (1 + self.gamma)) ** (n - 1)

            # Matrix constants
            a0 = lambda_var**n * a0b
            b0 = lambda_var**n * b0b
            a1 = lambda_var ** (n - 1) * a1b
            b1 = lambda_var ** (n - 1) * b1b

            # Linear System Matrix
            a = [
                [a0, b0, 0, 1],
                [b0, a0, lambda_var, 1],
                [a1, -b1, 1, 0],
                [b1, -a1, 1, 0],
            ]

            # Local solution
            s_vec = [self.y[segment], self.y[segment + 1], sp[segment], sp[segment + 1]]
            a, b, c, d = linalg.solve(a, s_vec)

            regression_parameters += [
                [[a, b, c, d], n, lambda_var, self.nodes[segment]]
            ]

        self.regression_parameters = regression_parameters
        self.fitted = True

        return regression_parameters

    def evaluate(self, input_vector: List[float]) -> List[float]:
        """
        Evaluates a fitted model in a set of points X.

        Parameters
        ----------
        input_vector : List[float]
            List of domain points to evaluate the fitted interpolating model.

        Returns
        -------
        List[float]
            List of the images of the fitted evaluated on the input domain X.

        Raises
        ------
        AttributeError
            If the attribute regression_parameters is None.
        """

        def get_segment_number(x: float, input_vector: List[float]) -> int:
            """
            Given an input list input_vector  of dimension N, returns the index of the
            N - 1 intervals [input_vector[i], input_vector[i+1]) to which x belongs to.

            Example:
            - x = 1.5
            - input_vector = [0, 1, 2, 3, 4]
            - Output: 1

            Parameters
            ----------
            x : float
                The value to evalate in the intervals defined by input_vector.
            input_vector : List[float]
                List of N points that define N - 1 intervals [input_vector[i],
                input_vector[i+1]). The value of the parameter x will be assigned to
                one of those intervals.

            Returns
            -------
            int
                The index of the window to which x belongs to.

            Raises
            ------
            AssertionError
                If x lies outside [input_vector[0], input_vector[-1]]
            """
            assert (x >= input_vector[0]) and (
                x <= input_vector[-1]
            ), f"x must belong to [{input_vector[0]}, {input_vector[-1]}]"

            n_segments = len(input_vector) - 1

            if x == input_vector[-1]:
                return n_segments - 1

            for inx in range(n_segments):
                if input_vector[inx] <= x and x < input_vector[inx + 1]:
                    return inx
            return -1

        if not self.fitted:
            raise AttributeError("regression_parameters not set. Run fit() first.")

        output = np.zeros(len(input_vector))

        for inx, x in enumerate(input_vector):
            segment_index = get_segment_number(x, self.nodes)
            segment_parameters = self.regression_parameters[segment_index]

            # Regression parameters
            aa = segment_parameters[0][0]  # type: ignore
            bb = segment_parameters[0][1]  # type: ignore
            cc = segment_parameters[0][2]  # type: ignore
            dd = segment_parameters[0][3]  # type: ignore

            nn = segment_parameters[1]
            ll = segment_parameters[2]
            x0 = segment_parameters[3]

            # Regression functions
            def xi(x):
                return (x - x0) / ll

            def test_new_reg(x):
                return (
                    ll**nn
                    * (
                        aa / (nn * (nn - 1)) * (xi(x) - (1 - self.gamma) / 2) ** nn
                        + bb / (nn * (nn - 1)) * (xi(x) - (1 + self.gamma) / 2) ** nn
                    )
                    + cc * ll * xi(x)
                    + dd
                )

            output[inx] = test_new_reg(x)

        return list(output)
