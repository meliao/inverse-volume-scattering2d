import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import sys
import h5py
import argparse
import os
import time
# from helpers.plot_solution import evaluate_sine_series


def load_field_in_hdf5(
    key: str, fp_out: str, idx_slice=slice(None), retries: int = 0
) -> np.ndarray:
    """Loads an individual field to the specified field in a given hdf5 file"""
    if not os.path.exists(fp_out):
        raise FileNotFoundError("Can't load field %s from %s" % (key, fp_out))
    if retries >= 10:
        raise IOError(f"(lfih) Couldn't open file after 10 tries")
    try:
        with h5py.File(fp_out, "r") as hf:
            data_loaded = hf[key][()]
            data = data_loaded[idx_slice]

        return data

    except BlockingIOError:
        print("File is blocked; on retry # %i", retries)
        time.sleep(30)
        return load_field_in_hdf5(key, fp_out, idx_slice, retries + 1)


def setup_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()
    parser.add_argument("--in_fp_preds", default="data/preds/scattering_objs_0.h5")
    parser.add_argument("--in_fp_gt", default="data/data_nu_4/measurements_0.h5")
    parser.add_argument("--sample_idx", type=int, default=4)
    parser.add_argument("--out_fp", default="examples/our_sample_sin_transform.mat")
    parser.add_argument("--plot_fp", default="examples/sine_invertability.png")
    parser.add_argument("--N", type=int, default=96)

    return parser.parse_args()

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

def sine_series(q: np.ndarray, X: np.ndarray, Y: np.ndarray, N: int):
    """
    Compute the coefficients of the 2D sine series given a signal q on a grid.

    Parameters:
    q (2D array): The signal on the grid.
    X (2D array): X coordinates of the grid.
    Y (2D array): Y coordinates of the grid.
    N (int): The size of the coefficient array (NxN).

    Returns:
    coefs (2D array): Coefficients of the sine series.
    """
    coefs = np.zeros((N, N))
    Mx, My = X.shape

    for j in range(N):
        for k in range(N):
            # Calculate the inner product <q, sin(j(x+pi/2))sin(k(y+pi/2))>
            sin_jx = np.sin((j + 1) * (X + np.pi / 2))
            sin_ky = np.sin((k + 1) * (Y + np.pi / 2))
            inner_product = np.sum(q * sin_jx * sin_ky)

            # Normalize the coefficient
            coefs[j, k] = (4 / (Mx * My)) * inner_product

    return coefs


def plot_sine_series_invertablility(
    preds: np.ndarray,
    preds_after_transform: np.ndarray,
    gt: np.ndarray,
    gt_after_transform: np.ndarray,
    save_fp: str,
) -> None:
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))

    im_0 = ax[0, 0].imshow(preds)
    plt.colorbar(im_0, ax=ax[0, 0])
    ax[0, 0].set_title("Preds")

    im_1 = ax[0, 1].imshow(preds_after_transform)
    plt.colorbar(im_1, ax=ax[0, 1])
    ax[0, 1].set_title("After transform")

    im_2 = ax[0, 2].imshow(np.abs(preds - preds_after_transform), cmap="hot")
    plt.colorbar(im_2, ax=ax[0, 2])
    ax[0, 2].set_title("Diffs")

    im_3 = ax[1, 0].imshow(gt)
    plt.colorbar(im_3, ax=ax[1, 0])
    ax[1, 0].set_title("Ground-Truth")

    im_4 = ax[1, 1].imshow(gt_after_transform)
    plt.colorbar(im_4, ax=ax[1, 1])
    ax[1, 1].set_title("After transform")

    im_5 = ax[1, 2].imshow(np.abs(gt - gt_after_transform), cmap="hot")
    plt.colorbar(im_5, ax=ax[1, 2])
    ax[1, 2].set_title("Diffs")

    fig.tight_layout()
    plt.savefig(save_fp)


def main(args: argparse.Namespace) -> None:

    print(f"Loading preds from {args.in_fp_preds}")

    preds = load_field_in_hdf5("q_cart", args.in_fp_preds)

    print(f"Loading ground-truth from {args.in_fp_gt}")

    gt = load_field_in_hdf5("q_cart", args.in_fp_gt)

    preds = preds[args.sample_idx]
    gt = gt[args.sample_idx]

    # Define the grid on [-pi/2, pi/2]^2
    x_vals = np.linspace(-np.pi / 2, np.pi / 2, preds.shape[1], endpoint=False)
    (X, Y) = np.meshgrid(x_vals, x_vals)

    print(f"Evaluating sine transform of size ({args.N}, {args.N})")

    coefs_pred = sine_series(preds, X, Y, args.N)
    coefs_gt = sine_series(gt, X, Y, args.N)

    # Save output to a matlab file
    print(f"Saving to output file: {args.out_fp}")

    out_dd = {"coefs_warm_start": coefs_pred, "coefs_ground_truth": coefs_gt}
    scipy.io.savemat(args.out_fp, out_dd)

    print("Inverting sine transform.")
    preds_out = evaluate_sine_series(coefs_pred, X, Y)
    gt_out = evaluate_sine_series(coefs_gt, X, Y)

    print(f"Making plot at {args.plot_fp}")

    plot_sine_series_invertablility(preds, preds_out, gt, gt_out, args.plot_fp)


if __name__ == "__main__":

    a = setup_args()
    main(a)
