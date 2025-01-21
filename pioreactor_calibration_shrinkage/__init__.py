# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib

import click
from pioreactor.calibrations import utils
from pioreactor.cli.calibrations import calibration
from pioreactor.cluster_management import get_workers_in_inventory
from pioreactor.logging import create_logger
from pioreactor.pubsub import get_from
from pioreactor.pubsub import post_into
from pioreactor.utils import managed_lifecycle
from pioreactor.utils.networking import resolve_to_address
from pioreactor.utils.timing import current_utc_datetime
from pioreactor.utils.timing import current_utc_timestamp
from pioreactor.whoami import get_unit_name
from pioreactor.whoami import UNIVERSAL_EXPERIMENT


def simple_prefix_hash(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()[:4]


def polynomial_features(x: list[float], degree: int):
    """
    Given a 1D array x, generate a 2D design matrix of shape (len(x), degree),
    where column j is x^j (j=degree-1-j..0).
    """
    import numpy as np

    x = np.asarray(x).flatten()
    n = x.shape[0]
    X = np.zeros((n, degree))
    for j in range(degree):
        X[:, j] = x ** (degree - 1 - j)
    return X


def objective_function(w, A, X_list, y_list, lambda_w, lambda_a):
    """
    Compute the regularized objective:
      sum_i ||y_i - A_i X_i w||^2 + lambda_w ||w||^2 + lambda_a sum_i ((A_i - 1)^2)
    """
    import numpy as np

    total = 0.0
    for i in range(len(X_list)):
        residual = y_list[i] - A[i] * X_list[i].dot(w)
        total += np.sum(residual**2)
    total += (lambda_w) * np.sum(w**2)
    total += (lambda_a) * np.sum((A - 1) ** 2)
    return total


def fit_model(X_list, y_list, d, lambda_w=0.1, lambda_a=0.1, max_iter=50, tol=1e-6, verbose=False):
    """
    Block-coordinate descent to solve:
      min_{w, A} sum_i ||y_i - A_i X_i w||^2 + lambda_w||w||^2 + lambda_a sum_i A_i^2.
    """
    import numpy as np

    N = len(X_list)  # number of subjects

    w = 0.05 * np.random.randn(d)

    A = np.ones(N)

    prev_obj = 1_000_000

    for it in range(max_iter):
        # === 1) Update A_i for each subject, given current w
        for i in range(N):
            Xi = X_list[i]
            yi = y_list[i]
            Xi_w = Xi.dot(w)  # shape (n,)
            denom = Xi_w.dot(Xi_w) + lambda_a
            A[i] = (Xi_w.dot(yi) + lambda_a) / denom

        # === 2) Update w, given all A_i
        # Summation terms
        # Weighted Gram matrix: sum_i (A_i^2 X_i^T X_i) + lambda_w I
        XtX = np.zeros((d, d))
        Xty = np.zeros(d)
        for i in range(N):
            Xi = X_list[i]
            Ai = A[i]
            XtX += (Ai**2) * Xi.T.dot(Xi)
            Xty += Ai * Xi.T.dot(y_list[i])

        # Add regularization to diagonal
        XtX += lambda_w * np.eye(d)

        # Solve for w
        # We'll do a stable solve or invert
        w = np.linalg.solve(XtX, Xty)
        # w = w / np.linalg.norm(w)

        # Check for convergence
        obj = objective_function(w, A, X_list, y_list, lambda_w, lambda_a)
        rel_change = abs(obj - prev_obj) / max(1.0, abs(prev_obj))
        if verbose:
            print(f"Iteration {it+1}, Obj={obj:.5f}, RelChange={rel_change:.5e}")
        if rel_change < tol:
            break
        prev_obj = obj

    return w, A


def prepare_data(data_from_workers: list, degree: int):
    import numpy as np

    X_list = []
    y_list = []
    for cal in data_from_workers:
        X_list.append(polynomial_features(cal["recorded_data"]["x"], degree))
        y_list.append(np.array(cal["recorded_data"]["y"]))

    return X_list, y_list


def green(text):
    return click.style(text, fg="green")


def main(device):
    logger = create_logger("shrink_calibrations", experiment=UNIVERSAL_EXPERIMENT)
    logger.info(f"Starting shrinkage of calibrations for {device}.")
    with managed_lifecycle(get_unit_name(), UNIVERSAL_EXPERIMENT, "shrink_calibrations"):
        # 1. get the device calibrations per worker
        data_from_workers = {}

        for worker in get_workers_in_inventory():
            try:
                calibrations = get_from(resolve_to_address(worker), f"/unit_api/calibrations/{device}").json()
            except Exception as e:
                print(e)
                continue
            if len(calibrations) == 0:
                continue

            click.clear()
            click.echo(green(f"Select which calibration to use from {worker}:"))
            click.echo()
            for cal in calibrations:
                click.echo(f"  •{'✓' if cal['is_active'] else ' '} {cal['calibration_name']}")

            click.echo()
            calibration_name = click.prompt(
                "Enter calibration name (SKIP to skip worker)",
                type=click.Choice([cal["calibration_name"] for cal in calibrations] + ["SKIP"]),
                show_choices=False,
            )
            if calibration_name == "SKIP":
                continue

            data_from_workers[worker] = next(
                cal for cal in calibrations if cal["calibration_name"] == calibration_name
            )

        N = len(data_from_workers)
        if N <= 1:
            logger.info(f"Not enough calibrations for device {device} to be able to be shrunk.")

        # 2. Get some metadata from the user.
        prefix = click.prompt(
            "Prefix for the new calibrations",
            type=str,
            default=f"shrunk-{simple_prefix_hash(current_utc_timestamp())}-",
        )

        while True:
            degree = click.prompt("Degree of polynomial to fit", type=int, default=3) + 1
            lambda_a = click.prompt(
                "Parameter for bringing calibrations closer to the average. Higher is more closeness.",
                type=float,
                default=5,
            )
            lambda_w = 0.05

            list_X, list_Y = prepare_data(data_from_workers.values(), degree)
            try:
                w_est, a_est = fit_model(
                    list_X,
                    list_Y,
                    degree,
                    lambda_a=lambda_a / N,
                    lambda_w=lambda_w / N / degree,
                    tol=1e-8,
                    verbose=True,
                    max_iter=500,
                )
            except Exception as e:
                print(e)

            click.echo()
            logger.info(f"{degree=}, {lambda_a=}")
            logger.info(f"kernel polynomial = {utils.curve_to_functional_form('poly', w_est)}")
            logger.info(f"worker-specific scalars = {a_est.tolist()}")
            if click.confirm("Okay with results?"):
                break
            click.echo()

        # distribute to workers
        for i, worker in enumerate(data_from_workers):
            cal = data_from_workers[worker]
            cal["curve_data_"] = (a_est[i] * w_est).tolist()
            cal["curve_type"] = "poly"
            cal["created_at"] = current_utc_datetime()
            cal["calibration_name"] = f"{prefix}{cal['calibration_name']}"
            try:
                r = post_into(resolve_to_address(worker), f"/unit_api/calibrations/{device}", json=cal)
                r.raise_for_status()
                logger.info(
                    f"Sent new {device} calibration `{cal['calibration_name']}` to {worker} successfully."
                )
            except Exception as e:
                print(e)


@calibration.command(name="shrinkage", help="shrink calibrations across the cluster")
@click.option("--device", required=True)
def shrink_calibrations(device: str) -> None:
    main(device)
