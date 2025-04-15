import torch


def f(x: torch.Tensor, y: torch.Tensor, a: float, b: float) -> torch.Tensor:
    return x * torch.sin(a * torch.pi * y) + y * torch.sin(b * torch.pi * x)


def f_grid(a: float, b: float, n: int) -> torch.Tensor:
    """
    Generate a grid of points in the range [0, 1] x [0, 1] and compute the function f at each point.
        :param a: parameter a for the function f
        :param b: parameter b for the function f
        :param n: number of points in each dimension
        :return: tensor of shape (n, n) containing the values of f at each point in the grid
    """
    x = torch.linspace(0, 1, n)
    y = torch.linspace(0, 1, n)
    X, Y = torch.meshgrid(x, y)
    return f(X, Y, a, b)


def derive_x(mat: torch.Tensor) -> torch.Tensor:
    """
    Compute the derivative of a matrix along axis x using finite differences, including borders.
        :param mat: tensor of shape (n, n)
        :return: tensor of shape (n, n) containing the derivatives
    """
    dx = 1 / (mat.shape[0] - 1)
    d_mat = torch.zeros_like(mat)
    d_mat[1:-1, :] = (mat[2:, :] - mat[:-2, :]) / (2 * dx)
    d_mat[0, :] = (mat[1, :] - mat[0, :]) / dx
    d_mat[-1, :] = (mat[-1, :] - mat[-2, :]) / dx

    return d_mat


def derive_y(mat: torch.Tensor) -> torch.Tensor:
    """
    Compute the derivative of a matrix along axis y using finite differences, including borders.
        :param mat: tensor of shape (n, n)
        :return: tensor of shape (n, n) containing the derivatives
    """
    dy = 1 / (mat.shape[1] - 1)
    d_mat = torch.zeros_like(mat)
    d_mat[:, 1:-1] = (mat[:, 2:] - mat[:, :-2]) / (2 * dy)
    d_mat[:, 0] = (mat[:, 1] - mat[:, 0]) / dy
    d_mat[:, -1] = (mat[:, -1] - mat[:, -2]) / dy

    return d_mat


def visu(mat: torch.Tensor) -> None:
    """
    Visualize the given matrix using matplotlib.
        :param mat: tensor of shape (n, n) to visualize
    """
    import matplotlib.pyplot as plt

    plt.imshow(mat.numpy(), cmap="hot", interpolation="nearest")
    plt.colorbar()
    plt.show()
