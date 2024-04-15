from fully_connected import FullyConnected
import torch


def fully_connected_test():
    """
    Provides Unit tests for the FullyConnected autograd Function

    PROVIDED CONSTANTS
    ------------------
    TOL (float): The error tolerance for the backward mode. If the error >= TOL, then is_correct is false
    DELTA (float): The difference parameter for the finite difference computations
    X (Tensor): of size (48 x 2), the inputs
    W (Tensor): of size (2 x 72), the weights
    B (Tensor): of size (72), the biases

    Returns
    -------
    is_correct (boolean): True if and only iff FullyConnected passes all unit tests
    err (Dictionary): with the following keys
                    1. dzdx: the error between the analytical and numerical gradients w.r.t X
                    2. dzdw (float): ... w.r.t W
                    3. dzdb (float): ... w.r.t B

    Note
    ----
    The error between arbitrary tensors x and y is defined here as the maximum value of the absolute difference between
    x and y.
    """
    # %%% DO NOT EDIT BELOW %%%%
    dataset = torch.load("fully_connected_test.pt")
    X = dataset["X"]
    W = dataset["W"]
    B = dataset["B"]
    TOL = dataset["TOL"]
    DELTA = dataset["DELTA"]
    full_connected = FullyConnected.apply
    # %%% DO NOT EDIT ABOVE
    X.requires_grad = True
    W.requires_grad = True
    B.requires_grad = True

    Y = FullyConnected.apply(X, W, B)
    Z = Y.mean()
    Z.backward()

    dzdx_analytical = X.grad
    dzdw_analytical = W.grad
    dzdb_analytical = B.grad

    with torch.no_grad():
        dzdx_num = torch.zeros_like(X)
        for i in range(X.size(0)):
            for j in range(X.size(1)):
                x_plus = X.clone()
                x_plus[i, j] += DELTA
                x_minus = X.clone()
                x_minus[i, j] -= DELTA
                fx_plus = FullyConnected.apply(x_plus, W, B)
                fx_minus = FullyConnected.apply(x_minus, W, B)
                dzdx_num[i, j] = (fx_plus - fx_minus).mean() / (2 * DELTA)

        dzdw_num = torch.zeros_like(W)
        for i in range(W.size(0)):
            for j in range(W.size(1)):
                w_plus = W.clone()
                w_plus[i, j] += DELTA
                w_minus = W.clone()
                w_minus[i, j] -= DELTA
                fx_plus = FullyConnected.apply(X, w_plus, B)
                fx_minus = FullyConnected.apply(X, w_minus, B)
                dzdw_num[i, j] = (fx_plus - fx_minus).mean() / (2 * DELTA)

        dzdb_num = torch.zeros_like(B)
        for i in range(B.size(0)):
            b_plus = B.clone()
            b_plus[i] += DELTA
            b_minus = B.clone()
            b_minus[i] -= DELTA
            fx_plus = FullyConnected.apply(X, W, b_plus)
            fx_minus = FullyConnected.apply(X, W, b_minus)
            dzdb_num[i] = (fx_plus - fx_minus).mean() / (2 * DELTA)

    err_dzdx = torch.max(torch.abs(dzdx_analytical - dzdx_num))
    err_dzdw = torch.max(torch.abs(dzdw_analytical - dzdw_num))
    err_dzdb = torch.max(torch.abs(dzdb_analytical - dzdb_num))

    is_correct = (err_dzdx < TOL) and (err_dzdw < TOL) and (err_dzdb < TOL) and torch.autograd.gradcheck(FullyConnected.apply, (X, W, B), eps=DELTA, atol=TOL)
    err = {
        "dzdx": err_dzdx.item(),
        "dzdw": err_dzdw.item(),
        "dzdb": err_dzdb.item()
    }

    torch.save([is_correct, err], "fully_connected_test_results.pt")
    return is_correct, err


if __name__ == '__main__':
    tests_passed, errors = fully_connected_test()
    assert tests_passed
    print(errors)
