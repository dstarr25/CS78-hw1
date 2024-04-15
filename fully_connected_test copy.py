from fully_connected import FullyConnected
import torch

def compute_numerical_gradient(x, w, b, delta):
    """
    Computes the numerical gradients of the FullyConnected function using finite differences.
    """
    with torch.no_grad():
        dzdx_num = torch.zeros_like(x)
        for i in range(x.size(0)):
            for j in range(x.size(1)):
                x_plus = x.clone()
                x_plus[i, j] += delta
                x_minus = x.clone()
                x_minus[i, j] -= delta
                fx_plus = FullyConnected.apply(x_plus, w, b)
                fx_minus = FullyConnected.apply(x_minus, w, b)
                dzdx_num[i, j] = (fx_plus - fx_minus).mean() / (2 * delta)

        dzdw_num = torch.zeros_like(w)
        for i in range(w.size(0)):
            for j in range(w.size(1)):
                w_plus = w.clone()
                w_plus[i, j] += delta
                w_minus = w.clone()
                w_minus[i, j] -= delta
                fx_plus = FullyConnected.apply(x, w_plus, b)
                fx_minus = FullyConnected.apply(x, w_minus, b)
                dzdw_num[i, j] = (fx_plus - fx_minus).mean() / (2 * delta)

        dzdb_num = torch.zeros_like(b)
        for i in range(b.size(0)):
            b_plus = b.clone()
            b_plus[i] += delta
            b_minus = b.clone()
            b_minus[i] -= delta
            fx_plus = FullyConnected.apply(x, w, b_plus)
            fx_minus = FullyConnected.apply(x, w, b_minus)
            dzdb_num[i] = (fx_plus - fx_minus).mean() / (2 * delta)

    return dzdx_num, dzdw_num, dzdb_num

def fully_connected_test():
    dataset = torch.load("fully_connected_test.pt")
    X = dataset["X"]
    W = dataset["W"]
    B = dataset["B"]
    TOL = dataset["TOL"]
    DELTA = dataset["DELTA"]

    X.requires_grad = True
    W.requires_grad = True
    B.requires_grad = True

    Y = FullyConnected.apply(X, W, B)
    Z = Y.mean()
    Z.backward()

    dzdx_analytical = X.grad
    dzdw_analytical = W.grad
    dzdb_analytical = B.grad

    dzdx_numerical, dzdw_numerical, dzdb_numerical = compute_numerical_gradient(X, W, B, DELTA)

    err_dzdx = torch.max(torch.abs(dzdx_analytical - dzdx_numerical))
    err_dzdw = torch.max(torch.abs(dzdw_analytical - dzdw_numerical))
    err_dzdb = torch.max(torch.abs(dzdb_analytical - dzdb_numerical))

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
