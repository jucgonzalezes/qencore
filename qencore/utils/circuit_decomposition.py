from typing import Any, List, Union

import numpy as np


def orthogonal_basis() -> List[np.ndarray]:
    identity = np.array([[1, 0], [0, 1]])
    pauli_x = np.array([[0, 1], [1, 0]])
    pauli_y = np.array([[0, -1j], [1j, 0]])
    pauli_z = np.array([[1, 0], [0, -1]])

    return [identity, pauli_x, pauli_y, pauli_z]


def complete_basis() -> List[np.ndarray]:
    a_set = orthogonal_basis()
    basis_a = []

    for i in range(4):
        for j in range(4):
            basis_a.append(np.kron(a_set[i], a_set[j]))
    return basis_a


def frobenius_norm(m1: np.ndarray, m2: np.ndarray) -> float:
    return np.trace(np.matrix(m1).H @ np.matrix(m2))


def get_projection_coeffs(a: Any, basis: List[np.ndarray]) -> np.ndarray[Any, Any]:
    coeffs = np.ones(len(basis)) + 0 * 1j * np.ones(len(basis))

    for inx, matrix in enumerate(basis):
        a_dot_matrix = frobenius_norm(a, matrix)
        matrix_dot_matrix = frobenius_norm(matrix, matrix)

        coeff = complex(a_dot_matrix) / complex(matrix_dot_matrix)
        coeffs[inx] = coeff

    return coeffs


def expand_basis(
    coeffs: List[float], basis: List[np.ndarray]
) -> np.ndarray[Any, np.dtype[Any]]:
    return np.real(np.matrix(sum(coeffs[i] * basis[i] for i in range(len(basis)))).T)


def is_properly_decomposed(
    a: List[List[Union[float, int]]], res_coeffs: Any, basis_a: List[np.ndarray]
) -> bool:
    return (np.matrix(a) == expand_basis(res_coeffs, basis_a)).all()
