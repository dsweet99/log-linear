import numpy as np
import statsmodels.api as sm

class LogLinear:
    def __init__(self, tolerance=1e-4, max_iterations=1000):
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def fit(self, X, y, b_trace=False):
        sigma2 = np.ones(shape=data.y.shape)
        if b_trace:
            llf_trace = []
        llf_prev = -1e99
        alpha = 1
        params = None
        for i_iter in range(self.max_iterations):
            # TODO: why not 1/sigma?
            mod = sm.WLS(data.y, X, weights=1 / sigma2)
            fit = mod.fit()
            if b_trace:
                llf_trace.append(fit.llf)
            conv_check = abs(fit.llf - llf_prev) / abs(fit.llf)
            if conv_check < self.tolerance:
                if b_trace:
                    return llf_trace
                return fit.params[1], fit.tvalues[1]
            llf_prev = fit.llf
            if params is None:
                params = fit.params
            else:
                params += alpha * (fit.params - params)
                alpha = max(0.1, 0.9 * alpha)
            eps = data.y - X @ params
            log_eps2 = np.log(eps**2)
            mod_var = sm.OLS(log_eps2, X)
            fit_var = mod_var.fit()
            sigma2 = np.exp(fit_var.predict(X))
        assert (
            False
        ), f"Reached self.max_iterations = {self.max_iterations} llf = {fit.llf} conv_check = {conv_check}"
