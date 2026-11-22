import numpy as np
from .structure_function import structure_function, characteristic_timescale
from .dcf import calculate_dcf_centroid
from .bootstrap import bootstrap_timescale_uncertainty


def print_dcf_binning_info(meta, target_lag=None):
    """
    Print diagnostic metrics about the automatically generated lag grid.
    Surfaces structural limits like the Nyquist cadence and pair density.
    """
    print("\nDCF Lag Bin Configuration (Auto-Driven)")
    print("-" * 50)
    print(f"  Lag Search Range  : {meta['lag_min']:.2f} → {meta['lag_max']:.2f}")
    print(f"  Bin Count         : {meta['n_bins']}")
    print(f"  Bin Width         : {meta['bin_width']:.2f}")
    print(f"  Nyquist Limit     : {meta['min_bin_width']:.2f}")
    print(f"  Cadence (t1 / t2) : {meta['cadence_t1']:.2f} / {meta['cadence_t2']:.2f}")
    print(f"  Total Cross-Pairs : {meta['n_total_pairs']:,}")
    print("-" * 50)

    if target_lag is not None:
        target_lag_abs = abs(target_lag)
        nyquist_limit = meta['min_bin_width']
        
        # How many bins does this target lag cover?
        spans_bins = target_lag_abs / meta['bin_width']
        
        print("\nTarget Lag Resolution Analysis")
        print("." * 50)
        
        if target_lag_abs < nyquist_limit:
            # Hard physical constraint failure
            max_allowed_cadence = target_lag_abs / 2.0
            print(f"  ✗ Target lag ({target_lag:.2f}) is UNRESOLVABLE.")
            print(f"    Reason            : It falls below the dataset's Nyquist limit ({nyquist_limit:.2f}).")
            print(f"    Required Cadence  : ≤ {max_allowed_cadence:.2f} to resolve this feature.")
            print(f"    Current Cadence   : {max(meta['cadence_t1'], meta['cadence_t2']):.2f} (Sparser series constraint).")
            print(f"    Remedy            : No binning strategy can recover this signal. You need denser data.")
            
        elif spans_bins < 3.0:
            # Physically present but statistically fragile due to wide bins
            print(f"  △ Target lag ({target_lag:.2f}) is MARGINALLY RESOLVED.")
            print(f"    Reason            : The feature spans only {spans_bins:.1f} bins.")
            print(f"    Risk              : Signals can be lost or smeared out across adjacent bins.")
            print(f"    Remedy            : If pair counts permit, try manually increasing 'n_bins'.")
            
        else:
            # Golden zone
            print(f"  ✓ Target lag ({target_lag:.2f}) is WELL RESOLVED.")
            print(f"    Status            : Spans {spans_bins:.1f} bins with healthy Nyquist sampling overhead.")
            
    print("-" * 50)


def print_dcf_significance(
    lag_dcf,
    dcf_vals,
    p95,
    p99,
    zero_lag_threshold=0.4
):
    """
    Generalized Peak/Valley Signification Assessment with Zero-Lag fallback avoidance.
    Identifies the dominant correlation feature while protecting against zero-lag traps.

    Parameters
    ----------
    lag_dcf : ndarray
        Calculated lag bin centers.
    dcf_vals : ndarray
        Discrete Correlation Function values.
    p95 : tuple of ndarrays
        (lower_bound, upper_bound) empirical 95% bootstrap significance floors.
    p99 : tuple of ndarrays
        (lower_bound, upper_bound) empirical 99% bootstrap significance floors.
    zero_lag_threshold : float, optional
        If the strongest peak sits inside the zero-lag bin, a secondary neighboring 
        peak is selected instead if its absolute magnitude exceeds this fraction of 
        the zero-lag peak. Range [0.0, 1.0]. Lower values are more aggressive at 
        skipping zero-lag. Default is 0.4.
    """
    # Exclude checking elements returning NaN values
    valid_mask = ~np.isnan(dcf_vals)
    if not np.any(valid_mask):
        print("Error: No valid lag bins recovered from calculation.")
        return

    # Look for maximum correlation strength (magnitude)
    abs_dcf = np.abs(dcf_vals)
    peak_idx = np.nanargmax(abs_dcf)
    
    # ADVANCED GENERALIZATION: Zero-Lag Trap Safeguard
    #Calculating the robust grid step size
    bin_width = np.median(np.diff(lag_dcf)) 
    # If the strongest peak is present on the 0.00 bin (central bin), but there is a clear secondary 
    # structure outside zero, check if it's an autocorrelation artifact.
    
    if np.abs(lag_dcf[peak_idx]) <= (bin_width / 2.0) and len(lag_dcf) > 3:
        # Temporarily suppress the center-most bin to review neighboring signals
        center_mask = np.abs(lag_dcf) > bin_width
        
        if np.any(valid_mask & center_mask):
            alt_idx = np.nanargmax(np.where(center_mask, abs_dcf, -np.inf))
            # If the secondary signal structure is strong, choose it over the raw zero artifact
            if abs_dcf[alt_idx] > zero_lag_threshold * abs_dcf[peak_idx]:
                peak_idx = alt_idx
    #Extract values based on the final optimized peak index
    peak_lag = lag_dcf[peak_idx]
    peak_dcf = dcf_vals[peak_idx]
    peak_abs = abs(peak_dcf)
    
    p95_lower,p95_upper = p95
    p99_lower,p99_upper = p99

    
    
    sig95 = (peak_dcf > p95_upper[peak_idx]) or (peak_dcf < p95_lower[peak_idx])
    sig99 = (peak_dcf > p99_upper[peak_idx]) or (peak_dcf < p99_lower[peak_idx])    
     
    
    print("\nDCF Peak Significance Diagnostics")
    print("-" * 50)
    print(f"Estimated Lag  (Peak Lag)  = {peak_lag:.3f}")
    print(f"Estimated Correlation(r)   = {peak_dcf:.4f}")
    print()        
    print("\nDCF Normalisation Note")
    print("-" * 50)
    
    if peak_abs <= 1.0:
        print("  |DCF| ≤ 1.0 — amplitude within standard bounds.")
        print("  Interpreting peak magnitude and sign directly is valid.")
        
    elif 1.0 < peak_abs <= 1.5:
        print(f"  |DCF| = {peak_abs:.3f} — mildly exceeds ±1.")
        print("  This is physically expected and NOT an error. Three")
        print("  mechanisms cause EK88 DCF to exceed ±1:")
        print()
        print("  [1] Noise correction: sig = sqrt(Var − <err²>) < total std.")
        print("      Removing noise variance shrinks the denominator,")
        print("      inflating the DCF amplitude above 1.0. This is")
        print("      correct physics — noise does not carry correlation.")
        print()
        print("  [2] Amplitude asymmetry: if one series has larger")
        print("      intrinsic amplitude, the cross-product can exceed")
        print("      the product of the individual normalising sigmas.")
        print()
        print("  [3] Local vs global normalisation: EK88 uses global")
        print("      mean/std but evaluates pairs in local lag bins.")
        print("      Bins near signal extremes can push UDCF beyond ±1.")
        print()
        print("  ► The LAG LOCATION and SIGN are fully reliable.")
        print("  ► SIGNIFICANCE is judged against the bootstrap envelope,")
        print("    not the ±1 bound. A DCF value that exceeds the")
        print("    99% null envelope is highly significant.")
        
    elif 1.5 < peak_abs <= 5.0:
        print(f"  |DCF| = {peak_abs:.3f} — substantially exceeds ±1.")
        print("  Check for strong amplitude asymmetry between your series")
        print("  or sparse bins with fewer than ~20 pairs. The lag")
        print("  location remains valid; treat the amplitude cautiously.")
        
    else:
        print(f"  |DCF| = {peak_abs:.3f} — anomalously large.")
        print("  Likely causes: data quality issue, extreme outliers,")
        print("  or a bin with very few pairs driving a large mean UDCF.")
        print("  Inspect the raw pair distribution before interpreting.")
        
    print("-" * 50)
    print()
    print(f"  95% Bootstrap Bound   : [{p95_lower[peak_idx]:.3f}, {p95_upper[peak_idx]:.3f}] → {'[SIGNIFICANT]' if sig95 else '[NOT SIGNIFICANT]'}")
    print(f"  99% Bootstrap Bound   : [{p99_lower[peak_idx]:.3f}, {p99_upper[peak_idx]:.3f}] → {'[SIGNIFICANT]' if sig99 else '[NOT SIGNIFICANT]'}")
    print("-" * 50)
    
    
    
def print_timescale_analysis(label, tau, sf):
    """
    Extracts and nicely prints the physical characteristic timescale 
    for an asset panel report.
    """
    # Call your analytical function from structure_function.py
    timescale = characteristic_timescale(tau, sf)
    
    print(f"\nStructure Function Timescale Analysis [{label}]")
    print("-" * 50)
    if np.isnan(timescale):
        print("  Characteristic Timescale : NOT FOUND (SF did not cross 1.0)")
        print("  Interpretation          : Observations remain correlated across all logged scales.")
    else:
        print(f"  Characteristic Timescale : {timescale:.3f} time units")
        print("  Interpretation          : Beyond this window, data points become ")
        print("                            statistically independent (decorrelated).")
    print("-" * 50)   
    
    

def print_enhanced_timescale_report(label, time, values, err=None, n_bins=None):
    """Prints characteristic timescale with explicit bootstrap error bars."""
    # 1. Core calculation
    tau, sf, _, _ = structure_function(time, values, err=err, n_bins=n_bins)
    raw_timescale = characteristic_timescale(tau, sf)
    
    print(f"\nStructure Function Timescale Analysis [{label}]")
    print("-" * 60)
    if np.isnan(raw_timescale):
        print("  Characteristic Timescale : NOT FOUND (SF did not cross 1.0)")
    else:
        # Run the new bootstrap engine
        med, low_err, up_err = bootstrap_timescale_uncertainty(time, values, err, n_bins, n_boot=100)
        if np.isnan(med):
            print(f"  Characteristic Timescale : {raw_timescale:.3f} time units (Uncertainty calculation failed)")
        else:
            print(f"  Characteristic Timescale : {raw_timescale:.3f} time units")
            print(f"  Bootstrap 1-σ Error Bar  : -{low_err:.3f} / +{up_err:.3f} (Median: {med:.3f})")
    print("-" * 60)

def print_dcf_peak_vs_centroid(lag_mid, dcf_vals):
    """Prints a comparison between the absolute peak lag and the robust centroid lag."""
    peak, centroid = calculate_dcf_centroid(lag_mid, dcf_vals, threshold=0.80, zero_lag_threshold=0.4)
    print(f"\nDCF Signal Localization Analysis")
    print("-" * 60)
    print(f"  Absolute Peak Lag  : {peak:.3f} time units (Highest discrete bin)")
    print(f"  Robust Centroid Lag: {centroid:.3f} time units (Flux-weighted center-of-mass)")
    #Calculate localized peak misalignment
    lag = abs(peak - centroid)
    bin_width = np.median(np.diff(lag_mid))
    if lag > bin_width: 
        print("  △ Notice: Centroid deviates significantly from the peak. The peak may be noisy.")
    else:
        print("  ✓ Agreement: Centroid and peak are well-aligned, indicating a clean symmetric signal.")
    print("-" * 60)    
