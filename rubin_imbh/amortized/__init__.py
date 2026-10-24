"""Training sets for amortized (simulation-based) inference of AGN + host + redshift.

Survey-agnostic SED simulation comes from :func:`jaxsedfit.simulate`; this package adds the
DP2-specific parts: population priors, the empirical noise/cadence library, the AGN
variability model, and reduction of simulated light curves with the pipeline's statistics
(:mod:`rubin_imbh.varstats`), so simulated and observed summaries mean the same thing.
"""
