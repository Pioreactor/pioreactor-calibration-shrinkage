from pioreactor.cluster_management import get_workers_in_inventory
from pioreactor.pubsub import get_from
from pioreactor.utils.networking import resolve_to_address
from pioreactor.cli.calibrations import calibration
from pioreactor.calibrations import utils
import click


def polynomial_features(x: list[float], degree: int):
    """
    Given a 1D array x, generate a 2D design matrix of shape (len(x), degree),
    where column j is x^j (j=0..degree-1).
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
      sum_i ||y_i - A_i X_i w||^2 + lambda_w ||w||^2 + lambda_a sum_i (A_i^2)
    """
    import numpy as np
    total = 0.0
    for i in range(len(X_list)):
        residual = y_list[i] - A[i] * X_list[i].dot(w)
        total += np.sum(residual**2)
    total += (lambda_w) * np.sum(w**2)
    total += (lambda_a) * np.sum((A-1)**2)
    return total

def fit_model(X_list, y_list, d, lambda_w=0.1, lambda_a=0.1,
              max_iter=50, tol=1e-6, verbose=False):
    """
    Block-coordinate descent to solve:
      min_{w, A} sum_i ||y_i - A_i X_i w||^2 + lambda_w||w||^2 + lambda_a sum_i A_i^2.
    """
    import numpy as np
    N = len(X_list)  # number of subjects

    w = 0.05 * np.random.randn(d)

    A = np.ones(N)

    prev_obj = 1000000

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
        #w = w / np.linalg.norm(w)

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
    for calibration in data_from_workers:
        X_list.append(polynomial_features(calibration['recorded_data']['x'], degree))
        y_list.append(np.array(calibration['recorded_data']['y']))

    return X_list, y_list

def green(text):
    return click.style(text, fg="green")

def main(device):
    # 1. get the device calibrations per worker
    data_from_workers = {}

    for worker in get_workers_in_inventory():
        try:
            calibrations = get_from(resolve_to_address(worker), f'/unit_api/calibrations/{device}').json()
        except Exception as e:
            print(e)
            continue
        click.clear()
        click.echo(green(f"""Select which calibration to use from {worker}:"""))
        click.echo()
        for cal in calibrations:

            click.echo(f" •{cal['calibration_name']}, created at {cal['created_at']}")

        click.echo()
        calibration_name = click.prompt(
            "Enter calibration name (SKIP to skip worker):",
            type=click.Choice([cal["calibration_name"] for cal in calibrations] + ["SKIP"]),
        )
        if calibration_name == "SKIP":
            continue
        data_from_workers[worker] = cal

    N = len(data_from_workers)

    # 2. Get some metadata from the user.
    prefix = click.prompt("Prefix for the new fused calibrations", type=str, default="fused-")


    while True:
        degree = click.prompt("Degree of polynomial to fit", type=int, default=4) + 1
        lambda_a = click.prompt("Parameter for bringing calibrations closer to the average. Higher is more closeness.", type=float, default=5)
        lambda_w = 0.05

        list_X, list_Y = prepare_data(data_from_workers.values(), degree)
        try:
            w_est, a_est = fit_model(list_X, list_Y, degree, lambda_a=lambda_a/N, lambda_w=lambda_w/N/degree, tol=1e-8, verbose=True, max_iter=500)
        except Exception as e:
            print(e)

        print()
        print(utils.curve_to_functional_form("poly", w_est))
        print(a_est)
        print()

        # confirm with user

    # distribute to workers

@calibration.command(name="shrinkage", help="shrink calibrations across the cluster")
@click.option("--device", required=True)
def shrink_calibrations(device: str) -> None:
    main(device)
