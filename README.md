# qencore (Quantum-Enhanced Convex Regression)

 [![License: Apache 2.0](https://img.shields.io/github/license/saltstack/salt)](https://opensource.org/license/apache-2-0) [![CI](https://github.com/jucgonzalezes/qencore/actions/workflows/ci.yml/badge.svg)](https://github.com/jucgonzalezes/qencore/actions/workflows/ci.yml)
 

A Python package to perform convex interpolation using splines, Input Convex 
Neural Networks, and Variational Quantum Linear Solvers.

## Poetry environment setup

We use `poetry` to manage the project's environment. Make sure to have it 
into you machine to run the code. To install it, run:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```
And then, install and run the environment.

```bash
poetry install
poetry shell
```

## Example usage 

You can find three use examples in `examples`. 

### Example 1: Classic Convex Interpolation

The script allows to perform classical convex regression by classical means 
using splines interpolation, a multilayer perceptron, and an input convex
neural network. 

#### Running 

```sh
poetry run python ./examples/convex_interpolation.py [--gamma GAMMA] [--sigma SIGMA]
```

**Parameters**

- `gamma` (float) : Defines the shape of the interpolating basis. The case gamma = 1 is equivalent to convex Hermite interpolation.
- `sigma` (float) : Value of the regularization strength of the empirical slopes. It should be a
        float value in the range [0.1, 0.2]
        

### Example 2: Variational Quantum Linear Solver

The script allows to solve a given linear system using the VQLS routine introduced in 
Variational Quantum Linear Solver (2023) by Bravo-Prieto et. al. The script 

#### Running 

```sh
poetry run python ./examples/vqls.py [--config CONFIG_PATH]
```

**Parameter**

- `config` (str) : Path to config file. By default in `examples/configs/vqls_config.yaml`

The config file 

- `num_qubits` (int) : Number of qubits used in the Quantum Circuit. Must coincide with len(gate_set[0]).
- `gate_set` (List[List[int]]): List of the single operation decomposition of the operator A. 
- `coefficient_set` List[float]: List of coefficients of the decomposition of the operator A.
- `num_shots` (int) : Number of shots used to estimate the expectation values.


### 3. Circuit decomposition

The script decompose a matrix into a linear combination of tensor products of
the Pauli operators. Here, we prepare 
```tex
A = \begin{bmatrix} 1 & 2 & 0 & 1 \\\ 
2 & 1 & 1 & 1 \\\ 4 & -5 & 1 & 0 \\\ 5 & -4 & 1 & 0 \end{bmatrix}
```

 #### Running 

```sh
poetry run python ./examples/circuit_decomposition.py [--config CONFIG_PATH]
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE)
file for details.
