"""
A scaler class that uses the empirical cummulative distribution function.

Author: Ilias Bilionis
"""

__all__ = ["ECDFScaler"]


import numpy as np
from numpy.typing import NDArray
import statsmodels.distributions.empirical_distribution as edf
from scipy.interpolate import interp1d
import scipy.stats as st
import tqdm


class ECDFScaler:
    """A class that scales by using the empirical CDF of the data."""

    def __init__(self):
        self._data = None
        self._ecdf = None
        self._iecdf = None
        self._ndim = None
        self._norm_rv = st.norm()

    def fit(self, data : NDArray) -> "ECDFScaler":
        """Compute the ECDF of the data to be used for later scaling.
        
        I got the idea from this question:
        https://stackoverflow.com/questions/44132543/python-inverse-empirical-cumulative-distribution-function-ecdf
        """
        self._data = data
        self._ndim = data.ndim
        self._ecdf = []
        self._iecdf = []
        if self._ndim == 1:
            data = data[:, None]
        for col in tqdm.tqdm(data.T):
            ecdf = edf.ECDF(col)
            col_changes = sorted(set(col).union([0.0, 1.0]))
            sample_edf_values_at_slope_changes = [ ecdf(item) for item in col_changes]
            inverted_edf = interp1d(sample_edf_values_at_slope_changes, col_changes)
            self._ecdf.append(ecdf)
            self._iecdf.append(inverted_edf)
            self._col_changes = col_changes
        return self

    def _flatten(self, out):
        """Flatten the output if needed."""
        return out.flatten() if self._ndim == 1 else out

    def transform(self, data_original : NDArray) -> NDArray:
        """Scales original data to R"""
        return self._flatten(
            np.array(
            [F(col)
             for F, col in zip(self._ecdf, data_original.T)]
            ).T
        )

    def inverse_transform(self, data_scaled : NDArray) -> NDArray:
        if self._ndim == 1:
            data_scaled = data_scaled[:, None]
        return self._flatten(
            np.array(
            [iF(col)
             for iF, col in zip(self._iecdf, data_scaled.T)]
            ).T
        )

    def fit_transform(self, data: NDArray) -> NDArray:
        """Fit to data, then transform it."""
        return self.fit(data).transform(data)