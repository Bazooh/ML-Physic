import random
import torch
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

from utils import f_grid, visu


def simulate_u_grid(a: float, b: float, n: int) -> torch.Tensor:
    """
    Solve -Δu = f(x,y) using finite difference on a grid of n² including borders.
        :param a: parameter a for the function f
        :param b: parameter b for the function f
        :param n: number of points (including boundaries)
        :return: solution u as (n, n) tensor (includes boundary values set to 0)
    """
    n_inner = n - 2

    b_vec = f_grid(a, b, n)[1:-1, 1:-1].flatten().numpy() / (n - 1) ** 2
    A = torch.zeros((n_inner**2, n_inner**2))

    def idx(i: int, j: int):
        return i * n_inner + j

    for i in range(n_inner):
        for j in range(n_inner):
            k = idx(i, j)
            A[k, k] = 4
            if i > 0:
                A[k, idx(i - 1, j)] = -1
            if i + 1 < n_inner:
                A[k, idx(i + 1, j)] = -1
            if j > 0:
                A[k, idx(i, j - 1)] = -1
            if j + 1 < n_inner:
                A[k, idx(i, j + 1)] = -1

    # Solve A u = b
    u_flat = spsolve(A.numpy(), b_vec)
    u_inner = torch.tensor(u_flat.reshape((n_inner, n_inner)), dtype=torch.float32)

    # Pad with zeros for Dirichlet boundary
    u = torch.zeros((n, n), dtype=torch.float32)
    u[1:-1, 1:-1] = u_inner

    return u


def create_dataset(
    a_range: tuple[float, float] = (-1, 1),
    b_range: tuple[float, float] = (-1, 1),
    n: int = 64,
    size: int = 1000,
) -> dict[tuple[float, float], torch.Tensor]:
    """
    Generate a dataset of solutions for the PDE -Δu = f(x,y) on a grid.
    Each solution corresponds to a pair of parameters (a, b) randomly sampled
    from the specified ranges.

        :param a_range: range for parameter a (min, max)
        :param b_range: range for parameter b (min, max)
        :param n: number of points (including boundaries) in the grid
        :param size: number of samples to generate
        :return: dictionary mapping (a, b) to the corresponding solution tensor
    """
    dataset: dict[tuple[float, float], torch.Tensor] = {}
    for _ in tqdm(range(size)):
        a, b = random.uniform(*a_range), random.uniform(*b_range)
        while (a, b) in dataset:
            a, b = random.uniform(*a_range), random.uniform(*b_range)
        dataset[a, b] = simulate_u_grid(a, b, n)
    return dataset


if __name__ == "__main__":
    # Example usage
    visu(simulate_u_grid(6.74, 6, 50))
