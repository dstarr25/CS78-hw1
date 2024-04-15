from generalized_logistic import GeneralizedLogistic
import torch


def generalized_logistic_test():
    """
    Provides Unit tests for the GeneralizedLogistic autograd Function

    PROVIDED CONSTANTS
    ------------------
    TOL1 (float): the  error tolerance for the forward mode. If the error >= TOL1, is_correct is false
    TOL2 (float): The error tolerance for the backward mode
    DELTA (float): The difference parameter for the finite differences computation
    X (Tensor): size (48 x 2) of inputs
    L, U, and G (floats): The parameter values necessary to compute the hyperbolic tangent (tanH) using
                        GeneralizedLogistic
    Returns:
    -------
    is_correct (boolean): True if and only if GeneralizedLogistic passes all unit tests
    err (Dictionary): with the following keys
                        1. y (float): The error between the forward direction and the results of pytorch's tanH
                        2. dzdx (float): the error between the analytical and numerical gradients w.r.t X
                        3. dzdl (float): ... w.r.t L
                        4. dzdu (float): ... w.r.t U
                        5. dzdg (float): .. w.r.t G
     Note
     -----
    The error between arbitrary tensors x and y is defined here as the maximum value of the absolute difference between
    x and y.
    """
    # %%%% DO NOT EDIT BELOW %%%
    dataset = torch.load("generalized_logistic_test.pt")
    X = dataset["X"]
    L = dataset["L"]
    U = dataset["U"]
    G = dataset["G"]
    TOL1 = dataset["TOL1"]
    TOL2 = dataset["TOL2"]
    DELTA = dataset["DELTA"]
    generalized_logistic = GeneralizedLogistic.apply
    # %%% DO NOT EDIT ABOVE %%%

    is_correct = True
    err = {}
    # Forward pass test
    Y = generalized_logistic(X, L, U, G)
    y_true = torch.tanh(X)
    err['y'] = torch.max(torch.abs(Y - y_true)).item()
    is_correct &= err['y'] < TOL1

    # Backward pass test
    Z = Y.mean()
    Z.backward()
    dzdx, dzdl, dzdu, dzdg = X.grad, L.grad.item(), U.grad.item(), G.grad.item()
    dzdy = torch.autograd.grad(Z, Y, create_graph=True)[0]
    # Numerical gradients
    dzdx_num = torch.zeros_like(X)

    with torch.no_grad():
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                x_plus = X.clone()
                x_plus[i, j] += DELTA
                x_minus = X.clone()
                x_minus[i, j] -= DELTA
                fx_plus = generalized_logistic(x_plus, L, U, G)
                fx_minus = generalized_logistic(x_minus, L, U, G)
                dzdx_num[i, j] = torch.sum(dzdy * (fx_plus - fx_minus).mean() / (2 * DELTA))

        dzdl_num = torch.sum(dzdy * (generalized_logistic(X, L + DELTA, U, G) - generalized_logistic(X, L - DELTA, U, G)).mean() / (2 * DELTA))
        dzdu_num = torch.sum(dzdy * (generalized_logistic(X, L, U + DELTA, G) - generalized_logistic(X, L, U - DELTA, G)).mean() / (2 * DELTA))
        dzdg_num = torch.sum(dzdy * (generalized_logistic(X, L, U, G + DELTA) - generalized_logistic(X, L, U, G - DELTA)).mean() / (2 * DELTA))

    err['dzdx'] = torch.max(torch.abs(dzdx - dzdx_num)).item()
    err['dzdl'] = abs(dzdl - dzdl_num).item()
    err['dzdu'] = abs(dzdu - dzdu_num).item()
    err['dzdg'] = abs(dzdg - dzdg_num).item()

    is_correct &= err['dzdx'] < TOL2
    is_correct &= err['dzdl'] < TOL2
    is_correct &= err['dzdu'] < TOL2
    is_correct &= err['dzdg'] < TOL2
    is_correct &= torch.autograd.gradcheck(GeneralizedLogistic.apply, (X, L, U, G), eps=DELTA, atol=TOL2)

    torch.save([is_correct, err], "generalized_logistic_test_results.pt")

    return is_correct, err

if __name__ == '__main__':
    test_passed, errors = generalized_logistic_test()
    print(errors)
    assert test_passed