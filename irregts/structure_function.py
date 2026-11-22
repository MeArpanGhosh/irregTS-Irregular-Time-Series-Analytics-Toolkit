import numpy as np
import warnings
from .utils import biweight_std, auto_log_bins

def structure_function(time, values, err=None, n_bins=None, lag_range=None):
    """
    Compute the normalized first order structure function of an irregular time series.

    Parameters
    ----------
    time : array-like
        Observation timestamps.

    values : array-like
        Measurement values.

    n_bins : int
        Number of lag bins.

    Returns
    -------
    tau_mid : ndarray
        Lag bin centers.

    sf : ndarray
        Structure function values.

    sf_err : ndarray
        Structure function uncertainties.

    n_pairs : ndarray
        Number of contributing pairs.
    """
    # 1. Enforce strict type float safety up front
    time   = np.asarray(time,   dtype=float)
    values = np.asarray(values, dtype=float)
    if err is not None:
        err = np.asarray(err, dtype=float)

    # 2. Check alignment constraint
    if len(time) != len(values):
        raise ValueError(
            f"Shape mismatch: time and values must have equal length, "
            f"got {len(time)} and {len(values)}"
        )
    
    #if len(time) <= 5:
    #   raise ValueError(f"Need at least 6 observations, got {len(time)}")
     
    # 3. Guard against NaNs/Infinities FIRST
    valid = np.isfinite(time) & np.isfinite(values)
    if err is not None:
        valid = valid & np.isfinite(err)
        
    # 4. Enforce structural minimum data requirement on the cleaned data
    n_valid = np.sum(valid)
    if n_valid <= 5:
        raise ValueError(
            f"Insufficient data: Need at least 6 finite observations after removing NaNs/Infinities, "
            f"but only found {n_valid} valid data points."
        )

    # 5. Extract cleaned, validated arrays safely
    time = time[valid]
    values = values[valid]
    if err is not None:
        err = err[valid]
    #Estimate the robust biweighted standard deviation
    sigma_bw = biweight_std(values)
    
    if sigma_bw > 1e-8:
        pass  # Tier 1: Normal execution path — biweight scale calculation succeeded.
    else:
        # Calculate standard deviation to distinguish between heavy clipping and a true flatline
        sigma_std = float(np.std(values))

        if sigma_std > 1e-8:
            # Tier 2: Biweight returned ~0 despite real variance existing.
            # Most common causes: 
            #   - >50% of values are identical (biweight design breakdown limit)
            #   - A single extreme outlier paired with an otherwise flat series
            #   - Heavy data quantization or low sensor resolution rounding
            warnings.warn(
                f"Biweight scale returned ~0 despite non-zero standard deviation "
                f"(σ_std={sigma_std:.4g}). This typically means >50% of values "
                f"are identical, or a single extreme outlier dominates the series. "
                f"Falling back to standard deviation for SF normalisation. "
                f"Verify data quality — SF results may be unreliable.",
                RuntimeWarning, 
                stacklevel=2
            )
            sigma_bw = sigma_std
        else:
            # Tier 3: Both biweight and std are ~0 — True constant flatline.
            # The structure function is mathematically undefined for constant data
            # (all pair differences are zero, normaliser denominator is zero).
            warnings.warn(
                f"Series has zero variance (true flatline — all values identical). "
                f"The structure function is mathematically undefined for constant "
                f"input. Applying a unit scale floor (σ=1.0) to prevent division "
                f"by zero, but SF output metrics will be completely meaningless. "
                f"Check your input telemetry data.",
                RuntimeWarning, 
                stacklevel=2
            )
            sigma_bw = 1.0

    # All unique pairs (i,j) with i > j
    idx_i, idx_j = np.triu_indices(len(time), k=1)
    tau_all = np.abs(time[idx_i] - time[idx_j])
    s_all = ( (values[idx_i] - values[idx_j]) ** 2 ) / sigma_bw**2

    # Auto log bins
    bins = auto_log_bins(tau_all, n_bins=n_bins, lag_range=lag_range)
    tau_mid = 0.5 * (bins[:-1] + bins[1:])

    sf = np.full(len(tau_mid), np.nan)
    sf_err = np.full(len(tau_mid), np.nan)
    n_pairs = np.zeros(len(tau_mid), dtype=int)

    for k in range(len(tau_mid)):

        mask = ( 
            (tau_all >= bins[k]) &
            (tau_all < bins[k + 1])
        )

        s_bin = s_all[mask]

        if len(s_bin) < 3:  # skipping poorly populated bins
            continue

        sf[k] = np.median(s_bin)

        sf_err[k] = (
            np.median(np.abs(s_bin - sf[k])) / # MAD
            np.sqrt(len(s_bin))
        )

        n_pairs[k] = len(s_bin)

    good = ~np.isnan(sf)

    return (
        tau_mid[good],
        sf[good],
        sf_err[good],
        n_pairs[good]
    )
    
    
def characteristic_timescale(tau, sf):
    """
    Find where the Structure Function crosses 1.0 using robust interpolation.
    
    Handles noisy oscillations around 1.0 and resolves multiple crossing points
    by isolating the primary global transition.

    Parameters
    ----------
    tau : array-like
        The lag bins associated with the Structure Function.
    sf : array-like
        Calculated Structure Function values.

    Returns
    -------
    float
        The estimated characteristic timescale, or np.nan if no crossing exists.
    """
    tau = np.asarray(tau, dtype=float)
    sf  = np.asarray(sf, dtype=float)

    # Filter out NaNs to ensure contiguous indexing operations
    valid = np.isfinite(tau) & np.isfinite(sf)
    if np.sum(valid) < 2:
        return np.nan
    tau, sf = tau[valid], sf[valid]

    # Vectorized check for crossings: sign change of (sf - 1.0)
    # Detects both upward and downward crossings
    sf_shifted = sf - 1.0
    crossings = np.where(sf_shifted[:-1] * sf_shifted[1:] <= 0)[0]

    if len(crossings) == 0:
        return np.nan

    # Robust Selection: If multiple crossings exist due to noise, 
    # take the median index crossing to stay close to the global equilibrium window.
    target_k = crossings[len(crossings) // 2]

    # Linear Interpolation calculation
    tau_low, tau_high = tau[target_k], tau[target_k + 1]
    sf_low, sf_high   = sf[target_k], sf[target_k + 1]

    # Prevent division by zero if two adjacent points have identical SF values exactly at 1.0
    if sf_high == sf_low:
        return tau_low

    slope = (sf_high - sf_low) / (tau_high - tau_low)
    timescale = tau_low + (1.0 - sf_low) / slope

    return timescale    
