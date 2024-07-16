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

from qiskit import QuantumCircuit


def example_v_circuit(n_qubits, parameters):
    # We define the V circuit using the Ansatz
    #
    # |0>--- Ry(θ1) ---•---•--- Ry(θ4)-------•--- Ry(θ7) ---
    # |0>--- Ry(θ2) ---|---|--- Ry(θ5)---•---|--- Ry(θ8) ---
    # |0>--- Ry(θ3) -------|--- Ry(θ6)---|---|--- Ry(θ9) ---
    #

    qc = QuantumCircuit(n_qubits)

    for qubit in range(n_qubits):
        qc.ry(parameters[qubit], qubit)

    qc.cz(0, 1)
    qc.cz(2, 0)

    for qubit in range(n_qubits):
        qc.ry(parameters[3 + qubit], qubit)

    qc.cz(1, 2)
    qc.cz(2, 0)

    for qubit in range(n_qubits):
        qc.ry(parameters[6 + qubit], qubit)

    return qc


def example_u_circuit(n_qubits):
    qc = QuantumCircuit(n_qubits)

    for i in range(n_qubits):
        qc.h(i)

    return qc


def example_a_circ(n_qubits, gate_set):

    circuit_list = [] * len(gate_set)

    for elm in gate_set:

        qc = QuantumCircuit(n_qubits)

        for qubit in range(n_qubits):
            if elm[qubit] == 0:
                qc.id(qubit)
            if elm[qubit] == 1:
                qc.z(qubit)

        circuit_list.append(qc)

    return circuit_list
