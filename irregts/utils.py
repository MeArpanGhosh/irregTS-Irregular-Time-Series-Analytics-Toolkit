import numpy as np
import warnings
from astropy.stats import biweight_scale


def biweight_std(x):
    """Robust biweighted standard deviation."""
    return biweight_scale(x, c=9.0)


def auto_log_bins(lags, n_bins=25, lag_range=None):
    """
    Generate logarithmically spaced lag bins for Structure Function analysis.
    
    Protects strictly against non-positive boundaries to prevent log10 execution failures.
    """
    if n_bins is None:
        n_bins = 25
        
    if lag_range is not None:
        # Pull manual limits and enforce absolute positive floor for log space
        lag_min = float(lag_range[0])
        lag_max = float(lag_range[1])
        if lag_min <= 0:
            # Fall back to a tiny fraction of max to preserve log scale geometry safely
            lag_min = lag_max * 1e-4
    else:
        # Automatic empirical density calculation
        positive_lags = lags[lags > 0]
        if len(positive_lags) == 0:
            # Complete safety fallback if no positive lags exist
            lag_min, lag_max = 0.1, 100.0
        else:
            lag_min = np.percentile(positive_lags, 5)
            lag_max = np.percentile(lags, 95)
            
        # Double-check safety guard
        if lag_min <= 0:
            lag_min = lag_max * 1e-4

    # Always return a pure, clean logarithmic progression grid
    return np.logspace(np.log10(lag_min), np.log10(lag_max), int(n_bins) + 1)


def dcf_lag_bins(t1, t2, n_bins=None, lag_range=None, min_pairs_per_bin=10):
    """
    Generate linearly spaced DCF lag bins appropriate for Two unevenly sampled time series.
    """
    t1 = np.sort(np.asarray(t1, dtype=float))
    t2 = np.sort(np.asarray(t2, dtype=float))

    # --- 1. All signed cross-lags scalar definition ---
    n_total_pairs = len(t1) * len(t2)

    # --- 2. Minimum resolvable lag: Nyquist of sparser series ---
    cadence1 = np.median(np.diff(t1))
    cadence2 = np.median(np.diff(t2))
    min_bin_width = 2.0 * max(cadence1, cadence2)

    # --- 3. EXPLICIT MEMORY SAFEGUARD SAMPLING ---
    # Define a clean, deterministic sentinel state up front
    delta_sample = None
    
    # If either property needs automatic calculation, and it's a huge dataset,
    # we pull our 1,000,000 pair sample up front EXACTLY ONCE.
    if (lag_range is None or n_bins is None) and (n_total_pairs > 1_000_000):
        idx_t1 = np.random.choice(len(t1), min(len(t1), 1000), replace=False)
        idx_t2 = np.random.choice(len(t2), min(len(t2), 1000), replace=False)
        delta_sample = (t2[idx_t2, None] - t1[None, idx_t1]).ravel()

    # --- 4. Resolve Lag Range ---
    if lag_range is None:
        baseline1 = t1.max() - t1.min()
        baseline2 = t2.max() - t2.min()
        
        # Intelligently focus on the dense overlapping region
        abs_bound = 0.40 * min(baseline1, baseline2)
        abs_bound = max(abs_bound, 10.0 * min_bin_width)  # Floor safeguard
        lag_min, lag_max = -abs_bound, abs_bound
    else:
        lag_min, lag_max = float(lag_range[0]), float(lag_range[1])

    total_window = lag_max - lag_min

    # --- 5. Resolve Bin Count via Explicit Sentinel Inspection ---
    if n_bins is None:
        if n_total_pairs > 1_000_000:
            # Deterministic lookup: delta_sample is guaranteed to be available here
            # because it was evaluated in step 3. The inline fallback is a safe fallback.
            delta_calc = delta_sample if delta_sample is not None else (
                (t2[np.random.choice(len(t2), min(len(t2), 1000), replace=False), None] - 
                 t1[None, np.random.choice(len(t1), min(len(t1), 1000), replace=False)]).ravel()
            )
            n_pairs_in_window = int(np.sum((delta_calc >= lag_min) & (delta_calc <= lag_max)))
            
            # Scale sample count back up to represent true global density
            sampling_fraction = len(delta_calc) / n_total_pairs
            estimated_true_pairs = n_pairs_in_window / sampling_fraction
            n_bins_from_pairs = int(estimated_true_pairs / min_pairs_per_bin)
        else:
            # Small dataset: Safe to compute the exact distribution grid
            delta = (t2[:, None] - t1[None, :]).ravel()
            n_pairs_in_window = int(np.sum((delta >= lag_min) & (delta <= lag_max)))
            n_bins_from_pairs = int(n_pairs_in_window / min_pairs_per_bin)

        n_bins_from_nyquist = int(total_window / min_bin_width)
        
        # Balance pair statistics against physical Nyquist resolution limits
        n_bins = min(n_bins_from_pairs, n_bins_from_nyquist)
        n_bins = max(min(n_bins, 200), 20)   # Hard clamps

    bin_width = total_window / n_bins
    
    # --- 6. Warn if resolution is likely insufficient ---
    ratio = bin_width / min_bin_width
    
    if ratio < 1:
       warnings.warn(
        f"Bin width ({bin_width:.2f}) is finer than the "
        f"adopted Nyquist limit ({min_bin_width:.2f}). "
        f"Bins may be sparsely populated. "
        f"Consider reducing n_bins or increasing "
        f"min_pairs_per_bin.",
        RuntimeWarning
      )
    elif ratio > 2:
     
       warnings.warn(
        f"Bin width ({bin_width:.2f}) is {ratio:.1f}× larger "
        f"than the adopted Nyquist limit "
        f"({min_bin_width:.2f}). "
        f"Temporal features narrower than roughly "
        f"{bin_width:.1f} time units may be blurred. "
        f"Consider narrowing lag_range or increasing n_bins.",
        RuntimeWarning
      )
    
    """
    if bin_width < min_bin_width:
        warnings.warn(
            f"Bin width ({bin_width:.2f}) is below the Nyquist limit "
            f"({min_bin_width:.2f}). Consider reducing n_bins.",
            RuntimeWarning
        )
    """ 
    meta = {
        "lag_min": lag_min,
        "lag_max": lag_max,
        "n_bins": n_bins,
        "bin_width": bin_width,
        "min_bin_width": min_bin_width,
        "cadence_t1": cadence1,
        "cadence_t2": cadence2,
        "n_total_pairs": n_total_pairs,
    }

    return np.linspace(lag_min, lag_max, n_bins + 1), meta
    
    
def save_sf_to_csv(filepath, tau, sf, sf_err, n_pairs):
    """
    Save calculated Structure Function metrics into a cleanly formatted CSV.
    """
    header = "lag_bin_center,structure_function,sf_uncertainty,contributing_pairs"
    data = np.column_stack((tau, sf, sf_err, n_pairs))
    
    np.savetxt(
        filepath, 
        data, 
        delimiter=",", 
        header=header, 
        comments="", 
        fmt=["%.6f", "%.6f", "%.6f", "%d"]
    )
    print(f"✓ Structure Function data successfully exported to: {filepath}")


def save_dcf_to_csv(filepath, lag, dcf_vals, dcf_err, p95=None, p99=None):
    """
    Save Discrete Correlation Function values and their bootstrap significance 
    envelopes into a cleanly formatted CSV.
    """
    # Base columns
    header = "lag_bin_center,dcf_value,dcf_uncertainty"
    columns = [lag, dcf_vals, dcf_err]
    formats = ["%.6f", "%.6f", "%.6f"]
    
    # Dynamically append bootstrap bounds if they exist
    if p95 is not None:
        p95_lower, p95_upper = p95
        header += ",p95_bound_lower,p95_bound_upper"
        columns.extend([p95_lower, p95_upper])
        formats.extend(["%.6f", "%.6f"])
        
    if p99 is not None:
        p99_lower, p99_upper = p99
        header += ",p99_bound_lower,p99_bound_upper"
        columns.extend([p99_lower, p99_upper])
        formats.extend(["%.6f", "%.6f"])
        
    data = np.column_stack(columns)
    
    np.savetxt(
        filepath, 
        data, 
        delimiter=",", 
        header=header, 
        comments="", 
        fmt=formats
    )
    print(f"✓ Discrete Correlation Function data successfully exported to: {filepath}")     
    
    
    

