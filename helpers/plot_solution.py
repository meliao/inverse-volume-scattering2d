import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import sys
import h5py
import argparse


def setup_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()
    parser.add_argument("--reconstruction_fp", default="examples/test_result.mat")
    parser.add_argument("--reference_fp", default="examples/test_data.mat")
    parser.add_argument(
        "--plot_fp_reconstructions", default="examples/reconstructions.png"
    )
    parser.add_argument("--plot_fp_errors", default="examples/errors.png")

    return parser.parse_args()


def plot_reconstructions(x: np.ndarray, q_true: np.ndarray, save_fp: str) -> np.ndarray:
    """x has shape (N_freqs, N_x, N_x).
    This function makes a column of N_freqs plots.

    Returns the relative L2 errors
    """
    n = x.shape[0]
    fig, ax = plt.subplots(n, 3, figsize=(15, 5 * n))
    rel_l2_error_vals = np.empty((n,))

    for i in range(n):

        rel_l2_error = np.linalg.norm(x[i] - q_true) / np.linalg.norm(q_true)
        im_i = ax[i, 0].imshow(x[i])
        ax[i, 0].set_title(f"Preds, i={i}")
        plt.colorbar(im_i, ax=ax[i, 0])

        im_1 = ax[i, 1].imshow(q_true)
        ax[i, 1].set_title("G.T.")
        plt.colorbar(im_1, ax=ax[i, 1])

        im_2 = ax[i, 2].imshow(np.abs(x[i] - q_true), cmap="hot")
        ax[i, 2].set_title(f"Errors, RelL2={rel_l2_error}")
        plt.colorbar(im_2, ax=ax[i, 2])

        rel_l2_error_vals[i] = rel_l2_error

    fig.tight_layout()
    plt.savefig(save_fp)

    return rel_l2_error


def plot_errors(error_arr: np.ndarray, k_arr: np.ndarray, save_fp: str) -> None:

    fig, ax = plt.subplots()
    ax.plot(k_arr, error_arr, "-")

    ax.set_xlabel("Incident wave frequency")
    ax.set_ylabel("Relative L2 Error")

    fig.tight_layout()
    plt.savefig(save_fp)


def evaluate_sine_series(coefs, X, Y):
    """
    Evaluate the 2D sine series on a 2D uniform grid.

    Parameters:
    coefs (2D array): Coefficients of the sine series.
    X (2D array): X coordinates of the grid.
    Y (2D array): Y coordinates of the grid.

    Returns:
    q (2D array): The evaluated sine series on the grid.
    """
    q = np.zeros_like(X)
    m, n = coefs.shape

    for j in range(m):
        for k in range(n):
            q += (
                coefs[j, k]
                * np.sin((j + 1) * (X + np.pi / 2))
                * np.sin((k + 1) * (Y + np.pi / 2))
            )

    return q


def main(args: argparse.Namespace) -> None:

    print(f"Loading data from {args.reconstruction_fp}")
    data = scipy.io.loadmat(args.reconstruction_fp)

    print(f"Loading reference data from {args.reference_fp}")
    with h5py.File(args.reference_fp) as f:
        coefs = f["coefs"][:]
        freqs = f["kvh"][:]

    coefs = coefs.T
    print(f"Reference coefs has shape {coefs.shape}")

    _, n_iters = data["solution"].shape

    print(f"Found solution with {n_iters} different reconstructions.")

    x = np.empty(
        (
            n_iters,
            data["solution"][0, 0][0].shape[0],
            data["solution"][0, 0][0].shape[1],
        )
    )

    print(f"Computing reference q")
    x_vals = np.linspace(-np.pi / 2, np.pi / 2, x.shape[1], endpoint=False)

    (X, Y) = np.meshgrid(x_vals, x_vals)
    q_true = evaluate_sine_series(coefs, X, Y)

    for i in range(n_iters):
        x[i] = data["solution"][0, i][0]

    print("Plotting reconstructions")
    rel_l2_error_vals = plot_reconstructions(x, q_true, args.plot_fp_reconstructions)

    plot_errors(rel_l2_error_vals, freqs, args.plot_fp_errors)


if __name__ == "__main__":
    a = setup_args()
    main(a)
