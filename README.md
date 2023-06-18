# LogLinear

Fit a linear model with loglinear variance

    y ~ X@beta + sigma*eps
    log(sigma2) ~ X_sigma@beta_sigma

    The hope is to get smaller standard errors for beta by
     modeling the heteroscedasticity.

    See
      H. Goldstein, Heteroscedasticity and complex variation
      https://www.bristol.ac.uk/media-library/sites/cmm/migrated/documents/modelling-complex-variation.pdf
    and references therein.


## Usage

Try something like:

```
ll = LogLinear()
ll.fit(X, X_sigma, y)

print (ll.beta, ll.se_beta)
```

See `tests/test_log_linear.py` for a working example.
