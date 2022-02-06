"""Computes dyadically spaced frequencies, for Morlet wavelet spectra."""
import numpy as np


def get_dyadic_frequencies(f_min, f_max, num_channels):
    """
    Returns dyadically spaced frequencies.

    Parameters
    ----------
    f_{min, max} : float
        Minimum and maximum frequency in Hz
    num_channels : int
        Number of dydically spaced frequency channels

    Returns
    -------
    f_channels : ndarray
        Frequcy channels in Hz
    """
    period_min = 1.0 / f_max
    period_max = 1.0 / f_min
    periods = period_min * (
        2
        ** (
            (np.arange(num_channels) * np.log(period_max / period_min))
            / (np.log(2) * (num_channels - 1))
        )
    )
    f_channels = (1.0 / periods)[::-1]
    return f_channels
