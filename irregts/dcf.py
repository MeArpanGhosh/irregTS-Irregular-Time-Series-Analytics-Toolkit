import numpy as np
import warnings
from .utils import dcf_lag_bins

def dcf(
    t1, x1, t2, x2,
    err1=None,
    err2=None,
    n_bins=None,
    lag_range=None
):
    """
    Discrete Correlation Function (Edelson & Krolik 1988). This has been optimised for generalised scales.

    Cross-correlates two unevenly-sampled time series to find
    a time delay. For each lag bin Δτ, collects all pairs (i,j)
    where t2[j] - t1[i] ∈ [Δτ_lo, Δτ_hi] and computes the
    unbinned DCF estimator (EK88 eq. 5).

    Parameters
    ----------
    t1, x1, err1 : timestamps, values, errors for series 1
    t2, x2, err2 : timestamps, values, errors for series 2
    lag_bins     : edges of the cross-lag bins (same units as t1/t2)

    Returns
    -------
    lag_mid : bin centre lags
    DCF     : DCF value per bin
    DCF_err : uncertainty per bin
    
    With adaptive fallback normalization for noise-dominated series.
    """
    
    # Converting the data types to float
    t1,x1 = np.asarray(t1, dtype=float), np.asarray(x1, dtype=float)
    t2,x2 = np.asarray(t2, dtype=float), np.asarray(x2, dtype=float)
    
    # If no errors are passed, default to zero noise
    if err1 is None:
        err1 = np.zeros_like(x1)
    else:
        err1 = np.asarray(err1, dtype=float)

    if err2 is None:
        err2 = np.zeros_like(x2)
    else:
        err2 = np.asarray(err2, dtype=float)
    
    #Checking alignment and structural integrity constraints
    if len(t1) != len(x1) or len(t1) != len(err1) or len(t2) != len(x2) or len(t2) != len(err2):
       raise ValueError("Length mismatch: Timestamps, values, and errors must align within each time series.")
       
    if len(t1) < 3 or len(t2) < 3:
       raise ValueError(f"Insufficient data for correlation: Got {len(t1)} and {len(t2)} points (Minimum required is 3).")   
    
    mu1 = np.nanmean(x1)
    mu2 = np.nanmean(x2)
    
    # Calculate global sample variances
    var_x1 = np.var(x1, ddof=1)
    var_x2 = np.var(x2, ddof=1)
    
    # Calculate mean square errors 
    mean_err2_1 = np.mean(err1**2) 
    mean_err2_2 = np.mean(err2**2) 
    
    # --- PROTECTION GUARD: Noise-dominated variance correction safety ---
    if var_x1 > mean_err2_1:
        sigma_x1 = np.sqrt(var_x1 - mean_err2_1)
    else:
        warnings.warn(
            "Series 1 appears noise-dominated (measurement errors exceed sample variance). "
            "Falling back to total uncorrected variance for DCF normalization stability.",
            UserWarning
        )
        sigma_x1 = np.sqrt(var_x1)
        
    if var_x2 > mean_err2_2:
        sigma_x2 = np.sqrt(var_x2 - mean_err2_2)
    else:
        warnings.warn(
            "Series 2 appears noise-dominated (measurement errors exceed sample variance). "
            "Falling back to total uncorrected variance for DCF normalization stability.",
            UserWarning
        )
        sigma_x2 = np.sqrt(var_x2)

    # Ensure denominators can never be exactly zero (e.g. constant value inputs)
    if sigma_x1 == 0: sigma_x1 = 1.0
    if sigma_x2 == 0: sigma_x2 = 1.0
     
    # --- VECTORIZED PROCESSING ---
    lag_bins, meta = dcf_lag_bins(t1, t2, n_bins=n_bins, lag_range=lag_range)
    
    lag_mid  = 0.5 * (lag_bins[:-1] + lag_bins[1:])
    dcf_vals = np.full(len(lag_mid), np.nan)
    dcf_err  = np.full(len(lag_mid), np.nan)

    # --- VECTORIZED PROCESSING ---
    delta_tau = t2[np.newaxis, :] - t1[:, np.newaxis]   
    udcf_all  = (x1[:, np.newaxis] - mu1) * (x2[np.newaxis, :] - mu2) / (sigma_x1 * sigma_x2)

    for k in range(len(lag_mid)):
        lo, hi = lag_bins[k], lag_bins[k + 1]
        mask = (delta_tau >= lo) & (delta_tau < hi)
        udcf_arr = udcf_all[mask]
        
        M = len(udcf_arr)
        if M >= 10:
            dcf_vals[k] = np.mean(udcf_arr)
            variance = np.sum((udcf_arr - dcf_vals[k]) ** 2)
            dcf_err[k] = np.sqrt(max(variance / (M * (M - 1)), 0.0))
        else:
            dcf_vals[k] = np.nan
            dcf_err[k] = np.nan

    # Opting to return lag_mid, values, errors,meta. 
    
    return lag_mid, dcf_vals, dcf_err, meta
    
    
    
def calculate_dcf_centroid(lag_mid, dcf_vals, threshold=0.80, zero_lag_threshold=None):
    """
    Calculate the peak and flux-weighted centroid lag, with an optional
    safeguard to clear zero-lag autocorrelation or daily sampling traps.

    Parameters
    ----------
    lag_mid : ndarray
        Calculated lag bin centers.
    dcf_vals : ndarray
        Discrete Correlation Function values.
    threshold : float, optional
        Fractional limit of peak value to define the cluster window. Default is 0.80.
    zero_lag_threshold : float or None, optional
        If a float is provided (e.g., 0.4), skips a dominant peak at the zero bin 
        if a secondary neighboring structure exceeds this fraction of its amplitude.
        If None, standard global peak matching is used.
    """
    lag_mid, dcf_vals = np.asarray(lag_mid), np.asarray(dcf_vals)
    valid_mask = ~np.isnan(dcf_vals)
    
    if not np.any(valid_mask):
        return np.nan, np.nan
        
    # 1. Locate the absolute highest correlation peak with safeguard : Zero-Lag Trap Bypass 
    abs_dcf = np.abs(dcf_vals)    
    peak_idx = np.argmax(np.abs(dcf_vals))
    
    if zero_lag_threshold is not None and len(lag_mid) > 3:
        bin_width = np.median(np.diff(lag_mid))
        
        # Check if the highest mathematical peak sits inside the central zero bin
        if np.abs(lag_mid[peak_idx]) <= (bin_width / 2.0):
            center_mask = np.abs(lag_mid) > bin_width
            
            if np.any(valid_mask & center_mask):
                alt_idx = np.nanargmax(np.where(center_mask, abs_dcf, -np.inf))
                # If the secondary physical lag structure is prominent, select it
                if abs_dcf[alt_idx] > zero_lag_threshold * abs_dcf[peak_idx]:
                    peak_idx = alt_idx
    
    # Extract stable metrics based on the optimized index selection
    peak_val = dcf_vals[peak_idx]
    peak_lag = lag_mid[peak_idx]
    
    # 2. Isolate contiguous bins around the peak that sit above the threshold (e.g., 80% of peak)
    threshold_val = threshold * abs(peak_val)
    above_thresh = np.abs(dcf_vals) >= threshold_val
    
    # Find the contiguous cluster containing the peak
    # (prevents distant secondary peaks from corrupting the centroid)
    labeled_mask = np.zeros_like(above_thresh, dtype=bool)
    
    # Expand left from peak
    for i in range(peak_idx, -1, -1):
        if above_thresh[i]: labeled_mask[i] = True
        else: break
    # Expand right from peak
    for i in range(peak_idx, len(dcf_vals)):
        if above_thresh[i]: labeled_mask[i] = True
        else: break
        
    # 3. Compute flux-weighted center of mass: Centroid = Sum(Lag * DCF) / Sum(DCF)
    cluster_lags = lag_mid[labeled_mask]
    cluster_vals = np.abs(dcf_vals[labeled_mask])  # Use absolute values for weights
    
    if len(cluster_vals) == 0 or np.sum(cluster_vals) == 0:
        return peak_lag, peak_lag # Fallback to peak if calculation collapses
        
    centroid_lag = np.sum(cluster_lags * cluster_vals) / np.sum(cluster_vals)
    return peak_lag, centroid_lag    
