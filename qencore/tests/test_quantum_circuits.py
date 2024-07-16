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

import pytest

from qencore.quantum_circuits import VQLS
from qencore.utils.example_circuits import example_a_circ as a_circ
from qencore.utils.example_circuits import example_u_circuit as u_circuit
from qencore.utils.example_circuits import example_v_circuit as v_circuit


@pytest.fixture
def param_gate_set_list():

    # In the gate convention 0 = Id, 1 = Z on the corresponding qubit
    # IMPORTANT: The qiskit convention is that the the gates operate
    # from right to left.
    gate_set = [[0, 0, 0], [0, 0, 1]]  # [[Id_0, Id_1, Id_2], [Id_0, Id_1, Z_2]]
    a_circuit_list = a_circ(3, gate_set)
    a_gate_list = [circuit.to_gate() for circuit in a_circuit_list]

    return a_gate_list


@pytest.fixture
def param_coefficient_set():
    return [0.55, 0.45]


@pytest.fixture
def test_parameters(param_gate_set_list, param_coefficient_set):
    return 3, param_coefficient_set, param_gate_set_list


def test_problem(test_parameters):
    test_problem = VQLS(3, test_parameters[1], test_parameters[2])

    test_values = [
        3.002e00,
        4.452e00,
        2.278e00,
        1.389e00,
        1.283e00,
        2.213e00,
        3.625e00,
        -1.057e00,
        2.852e00,
    ]

    print(
        test_problem.cost_function(
            test_values,
            test_parameters[0],
            test_parameters[2],
            test_parameters[1],
            u_circuit,
            v_circuit,
        )
    )

    test_problem.optimize(u_circuit, v_circuit, n_shots=1_000)
