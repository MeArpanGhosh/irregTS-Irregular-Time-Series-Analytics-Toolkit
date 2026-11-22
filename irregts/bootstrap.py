import numpy as np
from .dcf import dcf
from .structure_function import structure_function, characteristic_timescale

def bootstrap_dcf_significance(
    t1, x1, t2, x2,
    err1=None,
    err2=None,
    meta=None,
    n_bins=None,
    lag_range=None,
    n_bootstrap=500
):

    """
    Bootstrap DCF significance by randomly shuffling
    time-series values to destroy intrinsic correlations.

    Returns:
    --------
    p95 : tuple of ndarrays (lower, upper)
        Two-tailed 95th percentile null envelope

    p99 : tuple of ndarrays (lower, upper)
        Two-tailed 99th percentile null envelope
       
    """
    
    if meta is not None:
        resolved_nbins = meta["n_bins"]
        resolved_range = (meta["lag_min"], meta["lag_max"])
        actual_bins    = resolved_nbins
    else:
        dummy_mid, _, _, tmp_meta = dcf(
            t1, x1, t2, x2,
            err1, err2,
            n_bins=n_bins,
            lag_range=lag_range
        )
        resolved_nbins = tmp_meta["n_bins"]
        resolved_range = (tmp_meta["lag_min"], tmp_meta["lag_max"])
            
    
    # Initialize using the structural shape calculation
    #dummy_mid, _, _, meta = dcf(t1, x1, t2, x2,
    #                            err1, err2, 
    #                            n_bins=n_bins, lag_range=lag_range)
    #resolved_range = (meta['lag_min'], meta['lag_max'])   
    #resolved_nbins = meta['n_bins']                         
    #actual_bins = len(dummy_mid)
    
    dcf_boot = np.full((n_bootstrap, actual_bins), np.nan)

    for b in range(n_bootstrap):
        x1_shuf = np.random.permutation(x1)
        x2_shuf = np.random.permutation(x2)

        _, dcf_b, _, _ = dcf(
            t1, x1_shuf, t2, x2_shuf,
            err1, err2,
            n_bins=resolved_nbins,
            lag_range=resolved_range
        )
        
        if len(dcf_b) != actual_bins: 
           raise RuntimeError(
               f"DCF returned {len(dcf_b)} bins but "
               f"{actual_bins} were expected."
           )        
        dcf_boot[b, :] = dcf_b   
        #n_valid = min(len(dcf_b), actual_bins)
        #dcf_boot[b, :n_valid] = dcf_b[:n_valid]
    
    #If a lag bin cannot survive a valid calculation in at least half of  
    #the simulated universes, we do not have enough statistical data 
    #density to establish a trustworthy significance baseline. Mask it out

    # 1. Quantify simulation data density across bins
    coverage = np.sum(~np.isnan(dcf_boot), axis=0)
    low_coverage = coverage < (0.5 * n_bootstrap)
    # Computing the two tailed quantile bounds along matching axes while ignoring NaNs
    
    
    p95_lower = np.nanpercentile(dcf_boot, 2.5,  axis=0)
    p95_upper = np.nanpercentile(dcf_boot, 97.5, axis=0)
    
    p99_lower = np.nanpercentile(dcf_boot, 0.5,  axis=0)
    p99_upper = np.nanpercentile(dcf_boot, 99.5, axis=0) 

    #Applying the generalised coverage mask
    p95_lower[low_coverage] = np.nan
    p95_upper[low_coverage] = np.nan
    p99_lower[low_coverage] = np.nan
    p99_upper[low_coverage] = np.nan

    return (p95_lower, p95_upper), (p99_lower, p99_upper)
    
    
    
def bootstrap_timescale_uncertainty(time, values, err=None, n_bins=None, n_boot=100, confidence=0.68):
    """
    Compute statistical uncertainty constraints on the characteristic timescale
    via bootstrap resampling with replacement. 
    
    Returns (median_timescale, lower_error, upper_error).
    """
    #boot_timescales = []
    #n_pts = len(time)
    
    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)
    if err is not None:
       err = np.asarray(err, dtype=float)
    
    # Pre-filter NaNs up front to avoid wasting bootstrap loops on empty rows
    valid = np.isfinite(time) & np.isfinite(values)
    if err is not None:
        valid = valid & np.isfinite(err)
    
    time_clean = time[valid]
    values_clean = values[valid]
    err_clean = err[valid] if err is not None else None
    
    n_clean = len(time_clean)
    if n_clean <= 5:
        return np.nan, np.nan, np.nan
        
    boot_timescales = []    

    for _ in range(n_boot):
        # 1. Resample indices with replacement
        boot_idx = np.random.choice(n_clean, size=n_clean, replace=True)
        
        t_b = time_clean[boot_idx]
        v_b = values_clean[boot_idx]
        e_b = err_clean[boot_idx] if err_clean is not None else None
        
        # 2.Sort chronologically by time to preserve TS geometry
        sort_order = np.argsort(t_b)
        
        
        v_b = v_b[sort_order]
        e_b = e_b[sort_order] if e_b is not None else None
        t_b = t_b[sort_order]
        
        # 3. Re-compute the structure function curve and crossing point
        try:
            tau_b, sf_b, _, _ = structure_function(t_b, v_b, err=e_b, n_bins=n_bins)
            timescale_b = characteristic_timescale(tau_b, sf_b)
            if not np.isnan(timescale_b):
                boot_timescales.append(timescale_b)
        except (ValueError, ZeroDivisionError):
            continue  # Skip rare degenerate matrix allocations safely
            
    if len(boot_timescales) < max(10, int(n_boot * 0.1)):
        return np.nan, np.nan, np.nan
        
    # 4. Extract asymmetric confidence intervals (e.g., 68% for 1-sigma bounds)
    boot_timescales = np.sort(boot_timescales)
    pct_lower = (1.0 - confidence) / 2.0
    pct_upper = 1.0 - pct_lower
    
    median_ts = np.median(boot_timescales)
    low_bound = np.percentile(boot_timescales, pct_lower * 100)
    high_bound = np.percentile(boot_timescales, pct_upper * 100)
    
    return median_ts, median_ts - low_bound, high_bound - median_ts    
    
    
    
    
def bootstrap_lag_uncertainties(
    t1, x1, err1,
    t2, x2, err2,
    meta,
    n_boot=1000,
    seed=None,
    threshold=0.80,
    zero_lag_threshold=None,
    use_rss=True
):
    """
    Estimate Peak and Centroid lag uncertainties using
    Flux Randomization (FR) and optional Random Subset
    Selection (RSS).

    Parameters
    ----------
    t1, x1, err1 : array-like
        Time series 1 timestamps, values, and errors.

    t2, x2, err2 : array-like
        Time series 2 timestamps, values, and errors.

    meta : dict
        Metadata returned by dcf().

    n_boot : int
        Number of bootstrap realizations.

    seed : int or None
        Random seed for reproducibility.

    threshold : float
        Threshold passed to calculate_dcf_centroid().

    zero_lag_threshold : float or None
        Passed directly to calculate_dcf_centroid().

    use_rss : bool
        If True, performs FR/RSS.
        If False, performs FR only.

    Returns
    -------
    dict
        Peak and centroid lag distributions and
        asymmetric confidence intervals.
    """

    rng = np.random.default_rng(seed)

    # --------------------------------------------------
    # Fixed lag geometry from empirical DCF
    # --------------------------------------------------
    n_bins = meta["n_bins"]
    lag_range = (meta["lag_min"], meta["lag_max"])

    # --------------------------------------------------
    # Convert inputs once
    # --------------------------------------------------
    t1 = np.asarray(t1, dtype=float)
    x1 = np.asarray(x1, dtype=float)
    err1 = np.asarray(err1, dtype=float)

    t2 = np.asarray(t2, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    err2 = np.asarray(err2, dtype=float)

    peak_dist = []
    centroid_dist = []

    # --------------------------------------------------
    # Bootstrap loop
    # --------------------------------------------------
    for b in range(n_boot):

        if use_rss:
            # --------------------------
            # RSS resampling
            # --------------------------
            idx1 = rng.choice(len(x1), len(x1), replace=True)
            idx2 = rng.choice(len(x2), len(x2), replace=True)
            
            
            order1 = np.argsort(t1[idx1])
            order2 = np.argsort(t2[idx2])
            
            
            t1_b = t1[idx1][order1]
            x1_b = x1[idx1][order1]
            err1_b = err1[idx1][order1]

            t2_b = t2[idx2][order2]
            x2_b = x2[idx2][order2]
            err2_b = err2[idx2][order2]

        else:
            t1_b, x1_b, err1_b = t1, x1, err1
            t2_b, x2_b, err2_b = t2, x2, err2

        # --------------------------
        # Flux Randomization
        # --------------------------
        x1_b = x1_b + rng.normal(0.0, err1_b)
        x2_b = x2_b + rng.normal(0.0, err2_b)

        # --------------------------
        # DCF realization
        # --------------------------
        lag_mid, dcf_b, _, _ = dcf(
            t1_b, x1_b,
            t2_b, x2_b,
            err1_b, err2_b,
            n_bins=n_bins,
            lag_range=lag_range
        )

        if np.all(np.isnan(dcf_b)):
            continue

        try:
            peak_lag_b, centroid_lag_b = calculate_dcf_centroid(
                lag_mid,
                dcf_b,
                threshold=threshold,
                zero_lag_threshold=zero_lag_threshold
            )

            if np.isfinite(peak_lag_b):
                peak_dist.append(peak_lag_b)

            if np.isfinite(centroid_lag_b):
                centroid_dist.append(centroid_lag_b)

        except Exception:
            continue

    # --------------------------------------------------
    # Sanity checks
    # --------------------------------------------------
    if len(peak_dist) < max(10, 0.05 * n_boot):
        raise RuntimeError(
            f"Only {len(peak_dist)} valid peak-lag realizations "
            f"out of {n_boot} bootstraps."
        )

    if len(centroid_dist) < max(10, 0.05 * n_boot):
        raise RuntimeError(
            f"Only {len(centroid_dist)} valid centroid-lag realizations "
            f"out of {n_boot} bootstraps."
        )

    peak_dist = np.asarray(peak_dist)
    centroid_dist = np.asarray(centroid_dist)

    # --------------------------------------------------
    # Confidence intervals
    # --------------------------------------------------
    peak_low, peak_high = np.nanpercentile(
        peak_dist,
        [16, 84]
    )

    centroid_low, centroid_high = np.nanpercentile(
        centroid_dist,
        [16, 84]
    )

    peak_med = np.nanmedian(peak_dist)
    centroid_med = np.nanmedian(centroid_dist)

    # --------------------------------------------------
    # Return structured results
    # --------------------------------------------------
    return {

        "peak": {
            "median": peak_med,
            "minus": peak_med - peak_low,
            "plus": peak_high - peak_med,
            "low": peak_low,
            "high": peak_high,
            "distribution": peak_dist
        },

        "centroid": {
            "median": centroid_med,
            "minus": centroid_med - centroid_low,
            "plus": centroid_high - centroid_med,
            "low": centroid_low,
            "high": centroid_high,
            "distribution": centroid_dist
        }

    }    
