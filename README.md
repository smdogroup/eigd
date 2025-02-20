# eigd
## Tools for computing eigenvector derivatives

The `eigd` package uses the adjoint method to compute the total derivative of functions that depend on the eigenvalues and eigenvectors of a generalized eigenvalue problem. The eigenvalue problem takes the form

$$A(x) \phi_{i} = \lambda_{i} B(x) \phi_{i}$$ 

for $i = 1,\ldots,N$, where the matrices $A$ and $B$ are square and depend on design variables $x$ and $B$ is positive definite. The eigenvectors are $B$ orthonormal.

The eigenvectors and eigenvalues can then be used to compute a function of interest as

$$f(\lambda_{1}, \ldots, \lambda_{N}, \phi_{1}, \ldots, \phi_{N})$$

`eigd` takes the derivative of the function of interest with respect to the design variables using the adjoint method.

## Installation

* Clone this repository, then enter the folder in the command line terminal.
* Enter `pip install -e .` within the `eigd` folder.

## Citation

If you're using `eigd` in your work, please cite our paper:

Li, B., Kennedy, G.J. Adjoint methods for computing derivatives of functions of eigenvectors using shift-and-invert preconditioning. Struct Multidisc Optim 68, 4 (2025). [https://doi.org/10.1007/s00158-024-03940-6](https://doi.org/10.1007/s00158-024-03940-6)

```
@article{Li2024,
	author = {Li, Bao and Kennedy, Graeme J.},
	date = {2024/12/23},
	doi = {10.1007/s00158-024-03940-6},
	isbn = {1615-1488},
	journal = {Structural and Multidisciplinary Optimization},
	number = {1},
	pages = {4},
	title = {Adjoint methods for computing derivatives of functions of eigenvectors using shift-and-invert preconditioning},
	url = {https://doi.org/10.1007/s00158-024-03940-6},
	volume = {68},
	year = {2024}}
```
