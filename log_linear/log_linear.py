import numpy as np
import statsmodels.api as sm


class LogLinear:
    def __init__(self, tolerance=1e-4, max_iterations=1000):
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def fit(self, X, X_sigma, y):
        sigma2 = np.ones(shape=y.shape)

        X_sigma = np.concatenate(
            (
                np.ones(shape=(X_sigma.shape[0], 1)),
                X_sigma,
            ),
            axis=1,
        )

        llf_trace = []
        llf_prev = -1e99
        alpha = 1
        params = None
        for i_iter in range(self.max_iterations):
            mod = sm.WLS(y, X, weights=1 / sigma2)
            fit = mod.fit()
            llf_trace.append(fit.llf)
            conv_check = abs(fit.llf - llf_prev) / abs(fit.llf)
            if conv_check < self.tolerance:
                self.fit_ = fit
                self.coef_ = fit.params
                self.cov_ = fit.cov_params()
                return llf_trace
            llf_prev = fit.llf
            if params is None:
                params = fit.params
            else:
                params += alpha * (fit.params - params)
                alpha = max(0.1, 0.9 * alpha)
            eps = y - X @ params

            log_eps2 = np.log(eps**2)
            mod_var = sm.OLS(log_eps2, X_sigma)
            fit_var = mod_var.fit()
            sigma2 = np.exp(fit_var.predict(X_sigma))
        assert False, f"Reached self.max_iterations = {self.max_iterations} llf = {fit.llf} conv_check = {conv_check}"
