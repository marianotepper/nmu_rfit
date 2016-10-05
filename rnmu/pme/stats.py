import numpy as np
from scipy._lib.six import callable, string_types, xrange
from scipy.special import gammaln, smirnov
from scipy.stats import distributions, kstest
from scipy.stats.stats import KstestResult


def log_nchoosek(n, k):
    return gammaln(n + 1) - gammaln(n - k + 1) - gammaln(k + 1)


def concentration_nfa(membership, mss, cutoff=3):
    n = len(membership)
    idx = membership > np.exp(-(cutoff ** 2))
    membership = membership[idx]
    d_min, _ = kstest(membership, 'uniform', alternative='less')
    pvalue = smirnov(len(membership), d_min)
    return (log_nchoosek(n, mss) + np.log(pvalue)) / np.log(10)


def meaningful(membership, mss, log_epsilon=0):
    return concentration_nfa(membership, mss) < log_epsilon


def weighted_kstest(rvs, weights, cdf, args=(), N=20, alternative='two-sided',
                    mode='approx'):
    """
    Perform the Kolmogorov-Smirnov test for goodness of fit.

    This performs a test of the distribution G(x) of an observed
    random variable against a given distribution F(x). Under the null
    hypothesis the two distributions are identical, G(x)=F(x). The
    alternative hypothesis can be either 'two-sided' (default), 'less'
    or 'greater'. The KS test is only valid for continuous distributions.

    Parameters
    ----------
    rvs : str, array or callable
        If a string, it should be the name of a distribution in `scipy.stats`.
        If an array, it should be a 1-D array of observations of random
        variables.
        If a callable, it should be a function to generate random variables;
        it is required to have a keyword argument `size`.
    cdf : str or callable
        If a string, it should be the name of a distribution in `scipy.stats`.
        If `rvs` is a string then `cdf` can be False or the same as `rvs`.
        If a callable, that callable is used to calculate the cdf.
    args : tuple, sequence, optional
        Distribution parameters, used if `rvs` or `cdf` are strings.
    N : int, optional
        Sample size if `rvs` is string or callable.  Default is 20.
    alternative : {'two-sided', 'less','greater'}, optional
        Defines the alternative hypothesis (see explanation above).
        Default is 'two-sided'.
    mode : 'approx' (default) or 'asymp', optional
        Defines the distribution used for calculating the p-value.

          - 'approx' : use approximation to exact distribution of test statistic
          - 'asymp' : use asymptotic distribution of test statistic

    Returns
    -------
    statistic : float
        KS test statistic, either D, D+ or D-.
    pvalue :  float
        One-tailed or two-tailed p-value.

    Notes
    -----
    In the one-sided test, the alternative is that the empirical
    cumulative distribution function of the random variable is "less"
    or "greater" than the cumulative distribution function F(x) of the
    hypothesis, ``G(x)<=F(x)``, resp. ``G(x)>=F(x)``.

    Examples
    --------
    >>> from scipy import stats

    >>> x = np.linspace(-15, 15, 9)
    >>> stats.kstest(x, 'norm')
    (0.44435602715924361, 0.038850142705171065)

    >>> np.random.seed(987654321) # set random seed to get the same result
    >>> stats.kstest('norm', False, N=100)
    (0.058352892479417884, 0.88531190944151261)

    The above lines are equivalent to:

    >>> np.random.seed(987654321)
    >>> stats.kstest(stats.norm.rvs(size=100), 'norm')
    (0.058352892479417884, 0.88531190944151261)

    *Test against one-sided alternative hypothesis*

    Shift distribution to larger values, so that ``cdf_dgp(x) < norm.cdf(x)``:

    >>> np.random.seed(987654321)
    >>> x = stats.norm.rvs(loc=0.2, size=100)
    >>> stats.kstest(x,'norm', alternative = 'less')
    (0.12464329735846891, 0.040989164077641749)

    Reject equal distribution against alternative hypothesis: less

    >>> stats.kstest(x,'norm', alternative = 'greater')
    (0.0072115233216311081, 0.98531158590396395)

    Don't reject equal distribution against alternative hypothesis: greater

    >>> stats.kstest(x,'norm', mode='asymp')
    (0.12464329735846891, 0.08944488871182088)

    *Testing t distributed random variables against normal distribution*

    With 100 degrees of freedom the t distribution looks close to the normal
    distribution, and the K-S test does not reject the hypothesis that the
    sample came from the normal distribution:

    >>> np.random.seed(987654321)
    >>> stats.kstest(stats.t.rvs(100,size=100),'norm')
    (0.072018929165471257, 0.67630062862479168)

    With 3 degrees of freedom the t distribution looks sufficiently different
    from the normal distribution, that we can reject the hypothesis that the
    sample came from the normal distribution at the 10% level:

    >>> np.random.seed(987654321)
    >>> stats.kstest(stats.t.rvs(3,size=100),'norm')
    (0.131016895759829, 0.058826222555312224)

    """
    if isinstance(cdf, string_types):
        cdf = getattr(distributions, cdf).cdf

    idx = np.argsort(rvs)
    vals = rvs[idx]
    hist = np.insert(weights[idx], 0, 0)
    hist = np.cumsum(hist)
    hist /= hist[-1]
    N = len(vals)
    cdfvals = cdf(vals, *args)

    # to not break compatibility with existing code
    if alternative == 'two_sided':
        alternative = 'two-sided'

    if alternative in ['two-sided', 'greater']:
        Dplus = (hist[1:] - cdfvals).max()
        if alternative == 'greater':
            return KstestResult(Dplus, distributions.ksone.sf(Dplus, N))

    if alternative in ['two-sided', 'less']:
        Dmin = (cdfvals - hist[:-1]).max()
        if alternative == 'less':
            return KstestResult(Dmin, distributions.ksone.sf(Dmin, N))

    if alternative == 'two-sided':
        D = np.max([Dplus, Dmin])
        if mode == 'asymp':
            return KstestResult(D, distributions.kstwobign.sf(D * np.sqrt(N)))
        if mode == 'approx':
            pval_two = distributions.kstwobign.sf(D * np.sqrt(N))
            if N > 2666 or pval_two > 0.80 - N*0.3/1000:
                return KstestResult(D, pval_two)
            else:
                return KstestResult(D, 2 * distributions.ksone.sf(D, N))