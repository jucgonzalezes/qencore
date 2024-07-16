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

import random
from typing import List, Optional

import qiskit
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.gate import Gate
from qiskit_aer import AerSimulator
from scipy.optimize import minimize


class VQLS:
    r"""
    Class implementation of the Variational Quantum Linear Solver based on
    introduced by Bravo-Prieto et al (2023).

    The method `VQLS.optimize` finds the optimal set of parameters :math:`theta`
    such that a given parametrized Ansatz circuit :math:`V(\theta)` applied to the
    state :math:`| 0 \rangle` prepares a quantum state :math:`| x \rangle`
    encoding a scaled solution to the linear system :math:`A x = b`. That is,

    .. math::
        :math:`V(\theta) | 0 \rangle \propto | x \rangle.`

    Clearly, because :math:`| x \rangle` is a quantum state, it is normalized,
    and like so it encodes a scale solution to the problem - that in general is
    not normalized.

    Parameters
    ----------
    num_qubits : int
        Number of qubits used in the main quantum circuits.
    coefficients_set: List[float]
        Coefficients of the decomposition of the matrix :math:`A` involved in the
        linear problem.
    gate_set: List[List[int]]
        A list of lists containing the information of the gates needed to prepare the
        operator :math:`A`. The corrent encoding is the following:

        - 0 : Id (The identity operator)
        - 1 : Z (The Pauli Z operator)

        The position of each operator inside the inner list represents the target
        qubit for the corresponding operation. For example [0, 1, 0] represents
        a circuit of 3 qubits where the second qubit is affected by a Pauli Z
        rotation and the others are affected by the Identity operator.

    Attributes
    ----------
    optimal_parameters : List[float]
        List of the optimal parameter for the Variational Quantum Circuit
        encoding the solution of the linear problem.
    optimal_circuit : qiskit.QuantumCircuit
        Circuit prepared with the provided variational Ansatz and `optimal_parameters`.
    __nfeval : int
        Counter for the optimization callback.

    Methods
    -------
    hadamard_test(controlled_u, n_qubits, real=True)
        Creates the circuit for the Hadamard test of the circuit `controlled_u`.
        Such circuit should be already controled with 1 qubit. If `real == True`
        it prepares the circuit to compute the real part of the expectation value
        :math:`\langle 0 left| U \right| 0 \rangle`, otherwise it prepares the
        circuit to compute the imaginary part.
    measure_expectation(qc, n_shots=150_000)
        Computes the expectation value of a given hadamard test circuit `qc` by
        simulating the circuit and computing :math:`P(0)- P(1)`, where :math:`P(0)`
        is the proportion of counts of the state :math:`| 0 \rangle` in the first
        qubit, and :math:`P(1)` the proportion of counts of the state :math:`| 1 \rangle`
        in the first qubit. The simulation runs for `n_shots`.
    cost_function(params, num_qubits, a_gate_list, a_coeffs, u, v, num_shots=150_000)
        Computes the cost function as indicated on Variational Quantum Linear Solver
        based on Bravo-Prieto, Carlos, et al. 'Variational quantum linear solver.'
        Quantum 7 (2023): 1188. It uses `params` to prepare :math:`V(\theta)` and
        returns a result estimated with `num_shots` shots.
    optimize(u, v, method="POWELL", ftol=1e-6, initial_parameters=None, n_shots=150_000)
        Optimizes the parameters of the variational circuit :math:`V(\theta)`such that
        they minimize the cost function for the given problem.
    """

    def __init__(
        self, num_qubits: int, coefficient_set: List[float], gate_set: List[List[int]]
    ) -> None:
        self.num_qubits = num_qubits
        self.coefficient_set = coefficient_set
        self.gate_set = gate_set
        self.optimal_parameters = None
        self.optimal_circuit = None
        self.__nfeval = 1

    def hadamard_test(
        self,
        controlled_u: qiskit.circuit.ControlledGate,
        n_qubits: int,
        real: bool = True,
    ) -> QuantumCircuit:
        """
        Implements a circuit to perform a Hadamard test by introducing
        an auxiliary qubit to which a Hadamard Gate is applied before
        and after the main circuit. If `real = False`, it introduces
        a S gate after the first hadamard to account for the imaginary
        part of the expectation value.
        """
        auxiliary_qubit = QuantumRegister(1, "a")
        main_circuit = QuantumRegister(n_qubits, "q")

        hadamard_circuit = QuantumCircuit(auxiliary_qubit, main_circuit)
        gate_sequence = [auxiliary_qubit[0]] + [
            main_circuit[inx] for inx in range(n_qubits)
        ]

        hadamard_circuit.h(auxiliary_qubit[0])
        hadamard_circuit.sdg(auxiliary_qubit[0]) if not real else None
        hadamard_circuit.append(controlled_u, gate_sequence)
        hadamard_circuit.h(auxiliary_qubit[0])

        return hadamard_circuit

    def measure_expectation(self, qc: QuantumCircuit, n_shots: int = 150_000) -> float:
        """
        Computes the expectation value of a given Hadamard Circuit by computing
        :math:`P(0) - P(1)` where :math:`P(0)` and :math:`P(1)` represent the
        probabilities of measuring the states :math:`| 0 \rangle` and :math:`| 1 \rangle`
        on the first qubit, respectively.
        """
        circuit_copy = qc.copy()
        c_reg = ClassicalRegister(1, "c_bit")
        circuit_copy.add_register(c_reg)

        circuit_copy.measure(0, c_reg[0])

        simulator = AerSimulator()
        new_circuit = transpile(circuit_copy, backend=simulator)
        job = simulator.run(new_circuit, shots=n_shots)
        result = job.result()
        counts = result.get_counts()

        if "0" not in counts.keys():
            counts.update({"0": 0})
        elif "1" not in counts.keys():
            counts.update({"1": 0})

        # Calculate expectation value of Z
        expectation_value = float(counts["0"] - counts["1"]) / float(n_shots)

        return expectation_value

    def cost_function(
        self,
        params: List[float],
        num_qubits: int,
        a_gate_list: List[Gate],
        a_coeffs: List[float],
        u: Gate,
        v: Gate,
        num_shots: int = 150_000,
    ) -> float:
        """
        Computes the __global__ cost function.

        The global cost functions writes:

            C = Sum[ cl cl* ( Beta_l,l' - 1/n Gamma_l,l')
                        , {[l, l'], 1, len(gate_set)}]

        """

        v_gate = v(3, params).to_gate()
        u_gate = u(3).to_gate()

        # --------------- PART I : HADAMARD TEST FOR BETA
        # The loop computes the product of the coefficients cl cl*, and
        # computes the expectation values using the corresponding tests.
        # - Beta_l,l' : It uses the standard Hadamard Test
        #
        #            Re[Beta_l,l'] = P(0) - P(1) (Without the S gate)
        #
        #            Im[Beta_l,l'] = P(0) - P(1) (With the S gate)
        #
        #              Beta_l,l' = <0|A_dg A|0>

        denominator = 0.0

        for i, left_gate in enumerate(a_gate_list):
            for j, right_gate in enumerate(a_gate_list):

                inner_circuit = QuantumCircuit(num_qubits)
                main_circuit = QuantumRegister(num_qubits, "q")
                inner_circuit.append(
                    v_gate, [main_circuit[inx] for inx in range(num_qubits)]
                )
                inner_circuit.append(
                    right_gate, [main_circuit[inx] for inx in range(num_qubits)]
                )
                inner_circuit.append(
                    left_gate.inverse(),
                    [main_circuit[inx] for inx in range(num_qubits)],
                )
                inner_circuit.append(
                    v_gate.inverse(), [main_circuit[inx] for inx in range(num_qubits)]
                )
                u_controlled = inner_circuit.control(1)

                hadamard_circuit = self.hadamard_test(u_controlled, num_qubits)

                beta_l_lp = self.measure_expectation(
                    hadamard_circuit, n_shots=num_shots
                )
                denominator += a_coeffs[i].conjugate() * a_coeffs[j] * beta_l_lp

        # --------------- PART II : SPECIAL HADAMARD TEST FOR GAMMA
        # The loop computes the product of the coefficients cl cl*, and
        # computes the expectation values using the corresponding tests.

        numerator = 0.0

        for i, left_gate in enumerate(a_gate_list):
            for j, right_gate in enumerate(a_gate_list):

                # Subcircuit 1
                su1 = QuantumCircuit(num_qubits)
                main_sub_circuit_1 = QuantumRegister(num_qubits, "q")
                su1.append(
                    v_gate, [main_sub_circuit_1[inx] for inx in range(num_qubits)]
                )
                su1.append(
                    right_gate, [main_sub_circuit_1[inx] for inx in range(num_qubits)]
                )
                su1.append(
                    u_gate.inverse(),
                    [main_sub_circuit_1[inx] for inx in range(num_qubits)],
                )

                su1_controlled = su1.control(1)

                hadamard_sub_circuit_1 = self.hadamard_test(su1_controlled, num_qubits)
                sub_expectation_1 = self.measure_expectation(
                    hadamard_sub_circuit_1, n_shots=num_shots
                )

                # Subcircuit 2
                su2 = QuantumCircuit(num_qubits)
                main_sub_circuit_2 = QuantumRegister(num_qubits, "q")
                su2.append(
                    u_gate, [main_sub_circuit_2[inx] for inx in range(num_qubits)]
                )
                su2.append(
                    left_gate.inverse(),
                    [main_sub_circuit_2[inx] for inx in range(num_qubits)],
                )
                su2.append(
                    v_gate.inverse(),
                    [main_sub_circuit_1[inx] for inx in range(num_qubits)],
                )

                su2_controlled = su2.control(1)

                hadamard_sub_circuit_2 = self.hadamard_test(su2_controlled, num_qubits)
                sub_expectation_2 = self.measure_expectation(
                    hadamard_sub_circuit_2, n_shots=num_shots
                )

                gamma_l_lp = sub_expectation_1 * sub_expectation_2
                numerator += a_coeffs[i].conjugate() * a_coeffs[j] * gamma_l_lp

        total_cost = (1 - float(numerator / denominator)) ** 2

        return total_cost

    def optimize(
        self,
        u: Gate,
        v: Gate,
        method: str = "POWELL",
        ftol: float = 1e-6,
        initial_parameters: Optional[List[float]] = None,
        n_shots: int = 150_000,
    ) -> None:
        def callback_f(x):
            print(
                f"- Epoch : {self.__nfeval}\n \
                \t x = {x}\n \
                \t cost = {self.cost_function(x, self.num_qubits, self.gate_set, self.coefficient_set, u, v)}\n"
            )
            self.__nfeval += 1

        if not initial_parameters:
            initial_parameters = [
                float(random.randint(0000, 3000)) / 1000 for i in range(0, 9)
            ]

        out = minimize(
            self.cost_function,
            x0=initial_parameters,
            args=(self.num_qubits, self.gate_set, self.coefficient_set, u, v, n_shots),
            method=method,
            callback=callback_f,
            options={"maxiter": 400, "disp": True, "ftol": ftol},
        )

        self.optimal_parameters = out.x
        self.optimal_circuit = v(self.num_qubits, self.optimal_parameters)
        print(out)

        self.__nfeval = 1
