import numpy as np

from qencore.utils.circuit_decomposition import (
    complete_basis,
    get_projection_coeffs,
    is_properly_decomposed,
)

# Define the A matrix
a0, b0 = 1.0, 2.0
a1, b1 = 4.0, 5.0
lambda_val = 1.0

A = [[a0, b0, 0, 1], [b0, a0, lambda_val, 1], [a1, -b1, 1, 0], [b1, -a1, 1, 0]]
print(f"Input matrix A:\n{np.matrix(A)}")

basis_a = complete_basis()  # Set the decomposition basis
res_coeffs = get_projection_coeffs(A, basis_a)  # Project into the desired basis
is_succesful = is_properly_decomposed(A, res_coeffs, basis_a)

print(f"Succesfully decomposed? {is_succesful}")
