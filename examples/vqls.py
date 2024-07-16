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

import sys

import yaml  # type: ignore
from absl import flags

from qencore.quantum_circuits import VQLS
from qencore.utils.example_circuits import example_a_circ as a_circ
from qencore.utils.example_circuits import example_u_circuit as u_circuit
from qencore.utils.example_circuits import example_v_circuit as v_circuit


def main(num_qubits, gate_set, coefficient_set, num_shots):

    a_circuit_list = a_circ(num_qubits, gate_set)
    a_gate_list = [circuit.to_gate() for circuit in a_circuit_list]

    test_problem = VQLS(3, coefficient_set, a_gate_list)
    test_problem.optimize(u_circuit, v_circuit, n_shots=num_shots)


FLAGS = flags.FLAGS
flags.DEFINE_string(
    "config", "./examples/configs/vqls_config.yaml", "Path to config file."
)

if __name__ == "__main__":
    flags.FLAGS(sys.argv)

    with open(FLAGS.config, "r") as file:
        config = yaml.safe_load(file)

    main(
        config["num_qubits"],
        config["gate_set"],
        config["coefficient_set"],
        config["num_shots"],
    )
