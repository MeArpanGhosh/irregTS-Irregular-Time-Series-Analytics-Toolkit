# irregts

**A Python toolkit for irregular time-series analysis using Structure Functions, the Discrete Correlation Function (DCF), and bootstrap significance testing.**

Designed for astronomers and time-domain scientists working with unevenly sampled light curves — such as those from ZTF, CRTS, or spectroscopic reverberation mapping campaigns — but applicable to any irregularly sampled time-series data.

---

## Features

| Module | Description |
|---|---|
| **Structure Function** | First-order structure function with logarithmic binning and robust biweight normalization |
| **DCF** | Discrete Correlation Function ([Edelson & Krolik 1988](https://ui.adsabs.harvard.edu/abs/1988ApJ...333..646E)) with automatic Nyquist-aware lag binning |
| **Bootstrap** | Monte Carlo significance envelopes (95% & 99%) and bootstrap uncertainty on timescales and lag centroids |
| **Visualization** | Publication-ready combined plots (structure functions + DCF with significance bands) |
| **Reporting** | Diagnostic printouts for binning configuration, timescale analysis, and peak vs. centroid lag |
| **I/O Utilities** | CSV export for structure function and DCF results |

### Key Design Choices

- **Automatic lag grid construction** — Nyquist-aware binning adapts to your data's cadence and baseline; no manual tuning required.
- **Noise-dominated variance guard** — Falls back gracefully when measurement errors exceed sample variance.
- **Memory-safe large surveys** — Stochastic pair sampling for datasets with >10⁶ cross-pairs.
- **Robust statistics** — Biweight scale normalization resistant to outliers; MAD-based uncertainties.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/MeArpanGhosh/irregts.git
cd irregts
```

(Optional) Create a virtual environment:

```bash
conda create -n irregts python=3.11
conda activate irregts
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install the package in editable mode:

```bash
pip install -e .
```

---

## Quick Start

```python
import irregts
import numpy as np

# Load your time-series data
time = np.loadtxt("timestamps.dat")
flux = np.loadtxt("fluxes.dat")
err  = np.loadtxt("errors.dat")

# Structure Function
tau, sf, sf_err, n_pairs = irregts.structure_function(time, flux, err=err)
tau_char = irregts.characteristic_timescale(tau, sf)

# Cross-correlation (DCF) between two bands
lag, dcf_vals, dcf_err, meta = irregts.dcf(t1, x1, t2, x2, err1=e1, err2=e2)
peak_lag, centroid_lag = irregts.calculate_dcf_centroid(lag, dcf_vals)

# Bootstrap significance
p95, p99 = irregts.bootstrap_dcf_significance(t1, x1, t2, x2, err1=e1, err2=e2, meta=meta)

# Publication-ready plot
irregts.plot_combined_analysis(
    tau_a, sf_a, sf_a_err,
    tau_b, sf_b, sf_b_err,
    lag, dcf_vals, dcf_err,
    label_a="zg", label_b="zr",
    peak_lag=peak_lag,
    characteristic_tau_a=tau_char_a,
    characteristic_tau_b=tau_char_b,
    p95=p95, p99=p99
)

# Diagnostics
irregts.print_dcf_binning_info(meta)
irregts.print_dcf_significance(dcf_vals, lag, p95, p99)
```

---

## Examples

Detailed worked examples are provided as Jupyter notebooks using real astronomical light curves:

| Notebook | Object | Data Source | Analysis |
|---|---|---|---|
| [`Example1-NGC5548.ipynb`](Example1-NGC5548.ipynb) | NGC 5548 | Spectroscopic RM | Hβ vs Continuum 5100Å cross-correlation |
| [`Example2-Mrk290.ipynb`](Example2-Mrk290.ipynb) | Mrk 290 | ZTF | Multi-band (zg, zr) photometric correlation |
| [`Example2-Mrk355.ipynb`](Example2-Mrk355.ipynb) | Mrk 355 | ZTF | Multi-band (zg, zr) photometric correlation |
| [`Example-Mrk486.ipynb`](Example-Mrk486.ipynb) | Mrk 486 | ZTF | Multi-band (zg, zr, zi) photometric correlation |

---

## Sample Data

The `sample_data/` directory contains ZTF and spectroscopic light curves for several AGN:

```
sample_data/
├── ngc5548/          # Hβ and Continuum 5100Å spectroscopic data
├── mrk290/           # ZTF zg, zr, zi photometry
├── mrk335/           # ZTF zg, zr, zi photometry
├── mrk486/           # ZTF zg, zr, zi photometry
├── mrk493/           # ZTF zg, zr, zi photometry (two epochs)
├── Mrk817/           # ZTF zg, zr, zi photometry
├── NGC4253/          # ZTF zg, zr, zi photometry
├── PG0026+129/       # ZTF zg, zr, zi photometry
├── PG0052+251/       # ZTF zg, zr, zi photometry
└── PG1307+085/       # ZTF zg, zr, zi photometry (two epochs)
```

---

## API Reference

### Core Functions

#### `irregts.structure_function(time, values, err=None, n_bins=None, lag_range=None)`
Compute the normalized first-order structure function of an irregular time series.

**Returns:** `(tau_mid, sf, sf_err, n_pairs)`

#### `irregts.characteristic_timescale(tau, sf)`
Find where the structure function crosses 1.0 (the characteristic variability timescale).

**Returns:** `float` — the timescale in the same units as `tau`, or `np.nan` if no crossing exists.

#### `irregts.dcf(t1, x1, t2, x2, err1=None, err2=None, n_bins=None, lag_range=None)`
Discrete Correlation Function (Edelson & Krolik 1988) with automatic Nyquist-aware binning.

**Returns:** `(lag_mid, dcf_vals, dcf_err, meta)`

#### `irregts.calculate_dcf_centroid(lag_mid, dcf_vals, threshold=0.80, zero_lag_threshold=None)`
Calculate the peak and flux-weighted centroid lag with optional zero-lag trap bypass.

**Returns:** `(peak_lag, centroid_lag)`

### Bootstrap Functions

#### `irregts.bootstrap_dcf_significance(t1, x1, t2, x2, ..., n_bootstrap=500)`
Monte Carlo null-hypothesis significance envelopes for the DCF.

**Returns:** `(p95, p99)` — each a tuple of `(lower_bound, upper_bound)` arrays.

#### `irregts.bootstrap_timescale_uncertainty(time, values, err=None, n_bins=None, n_boot=100)`
Bootstrap uncertainty on the characteristic timescale.

**Returns:** `(median_timescale, lower_error, upper_error)`

#### `irregts.bootstrap_lag_uncertainties(t1, x1, t2, x2, ..., n_boot=100)`
Bootstrap uncertainty on DCF peak and centroid lags.

### Visualization & Reporting

#### `irregts.plot_combined_analysis(...)`
Publication-ready 2×2 figure with structure functions and DCF + significance bands.

#### `irregts.print_dcf_binning_info(meta)` · `irregts.print_dcf_significance(...)` · `irregts.print_enhanced_timescale_report(...)` · `irregts.print_dcf_peak_vs_centroid(...)`
Diagnostic console output.

#### `irregts.save_sf_to_csv(filepath, ...)` · `irregts.save_dcf_to_csv(filepath, ...)`
Export results to CSV.

---

## Project Structure

```
irregts/
├── irregts/                 # Core package
│   ├── __init__.py          # Public API exports
│   ├── structure_function.py
│   ├── dcf.py
│   ├── bootstrap.py
│   ├── visualization.py
│   ├── reporting.py
│   └── utils.py
├── sample_data/             # Example light curves
├── Example1-NGC5548.ipynb   # Example notebooks
├── Example2-Mrk290.ipynb
├── Example2-Mrk355.ipynb
├── Example-Mrk486.ipynb
├── pyproject.toml
├── setup.py
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Citation

If you use `irregts` in your research, please cite:

```
Ghosh, A. (2026). irregts: Irregular Time-Series Analysis Toolkit.
GitHub repository: https://github.com/arpanghosh2000/irregts
```

The DCF implementation follows:
> Edelson, R. A. & Krolik, J. H. (1988). *The Discrete Correlation Function: A New Method for Analyzing Unevenly Sampled Variability Data.* The Astrophysical Journal, 333, 646. [ADS](https://ui.adsabs.harvard.edu/abs/1988ApJ...333..646E)

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
