def test_log_linear():
    import numpy as np
    from sklearn.linear_model import LinearRegression

    from log_linear.log_linear import LogLinear

    ll = LogLinear()
    lr = LinearRegression(fit_intercept=False)

    np.random.seed(17)
    n = 100
    k = 10
    k_sigma = 3
    delta = []
    for _ in range(100):
        X = np.random.normal(size=(n, k))
        X_sigma = np.random.normal(size=(n, k_sigma))

        beta_sigma_0 = 0.1 + 0.9 * np.random.uniform()
        beta_sigma = np.random.normal(size=(k_sigma,))
        log_sigma2 = beta_sigma_0 + X_sigma @ beta_sigma

        sigma = np.sqrt(np.exp(log_sigma2))
        eps = sigma * np.random.normal(size=(n,))
        beta = np.random.normal(size=(k,))
        y = X @ beta + eps

        ll.fit(X, X_sigma, y)
        lr.fit(X, y)
        delta.append(np.linalg.norm(beta - ll.coef_) - np.linalg.norm(beta - lr.coef_))

    delta = np.array(delta)
    mu = delta.mean()
    se = delta.std() / np.sqrt(len(delta))
    t = mu / se
    assert t < -2, (mu, se, t)
