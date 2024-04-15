import torch.autograd.gradcheck
from mean_squared_error import MeanSquaredError
import torch


def mean_squared_error_test():
    """
     Unit tests for the MeanSquaredError autograd Function.

    PROVIDED CONSTANTS
    ------------------
    TOL (float): the absolute error tolerance for the backward mode. If any error is equal to or
                greater than TOL, is_correct is false
    DELTA (float): The difference parameter for the finite difference computation
    X1 (Tensor): size (48 x 2) denoting 72 example inputs each with 2 features
    X2 (Tensor): size (48 x 2) denoting the targets

    Returns
    -------
    is_correct (boolean): True if and only if MeanSquaredError passes all unit tests
    err (Dictionary): with the following keys
                    1. dzdx1 (float): the  error between the analytical and numerical gradients w.r.t X1
                    2. dzdx2 (float): The error between the analytical and numerical gradients w.r.t X2
    Note
    -----
    The error between arbitrary tensors x and y is defined here as the maximum value of the absolute difference between
    x and y.
    """
    # %%% DO NOT EDIT BELOW %%%
    dataset = torch.load("mean_squared_error_test.pt")
    X1 = dataset["X1"]
    X2 = dataset["X2"]
    TOL = dataset["TOL"]
    DELTA = dataset["DELTA"]
    mean_squared_error = MeanSquaredError.apply
    # %%% DO NOT EDIT ABOVE %%%
    X1.requires_grad = True
    X2.requires_grad = True
    Y = mean_squared_error(X1, X2)
    Z = Y.mean()
    Z.backward()
    dzdx1_analytical = X1.grad
    dzdx2_analytical = X2.grad
    dzdx1_numerical = torch.zeros_like(X1)
    dzdx2_numerical = torch.zeros_like(X2)

    with torch.no_grad():
        for i in range(X1.size(0)):
            for j in range(X1.size(1)):
                x1_plus = X1.clone()
                x1_plus[i, j] += DELTA
                x1_minus = X1.clone()
                x1_minus[i, j] -= DELTA
                fx_plus = mean_squared_error(x1_plus, X2)
                fx_minus = mean_squared_error(x1_minus, X2)
                dzdx1_numerical[i, j] = (fx_plus - fx_minus).mean() / (2 * DELTA)
        
        for i in range(X2.size(0)):
            for j in range(X2.size(1)):
                x2_plus = X2.clone()
                x2_plus[i, j] += DELTA
                x2_minus = X2.clone()
                x2_minus[i, j] -= DELTA
                fx_plus = mean_squared_error(X1, x2_plus)
                fx_minus = mean_squared_error(X1, x2_minus)
                dzdx2_numerical[i, j] = (fx_plus - fx_minus).mean() / (2 * DELTA)
    
    err_dzdx1 = torch.max(torch.abs(dzdx1_analytical - dzdx1_numerical))
    err_dzdx2 = torch.max(torch.abs(dzdx2_analytical - dzdx2_numerical))
    
    is_correct = (err_dzdx1 < TOL) and (err_dzdx2 < TOL) and torch.autograd.gradcheck(mean_squared_error, (X1, X2), eps=DELTA, atol=TOL)
    err = {
        "dzdx1": err_dzdx1.item(),
        "dzdx2": err_dzdx2.item(),
    }

    torch.save([is_correct, err], "mean_squared_error_test_results.pt")

    return is_correct, err


if __name__ == '__main__':
    tests_passed, errors = mean_squared_error_test()
    assert tests_passed
    print(errors)
