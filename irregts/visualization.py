import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_combined_analysis(
    tau_a,
    sf_a,
    sf_a_err,
    tau_b,
    sf_b,
    sf_b_err,
    lag_dcf,
    dcf_vals,
    dcf_err,
    label_a='Series A',
    label_b='Series B',
    color_a='steelblue',
    color_b='darkorange',
    peak_lag=None,
    characteristic_tau_a=None,
    characteristic_tau_b=None,
    p95=None,
    p99=None
):

    fig = plt.figure(figsize=(16, 14), dpi=100)

    gs = gridspec.GridSpec(
        2,
        2,
        hspace=0.35,
        wspace=0.32,
        left=0.12,
        right=0.97,
        top=0.95,
        bottom=0.10
    )

    plt.rc('font', size=16)
    plt.rc('axes', labelsize=20)
    plt.rc('legend', fontsize=14)

    # Panel 1
    ax1 = fig.add_subplot(gs[0, 0])

    ax1.errorbar(
        tau_a,
        sf_a,
        yerr=sf_a_err,
        fmt='o',
        color=color_a,
        ms=6,
        elinewidth=1.2,
        capsize=3,
        label=label_a
    )

    ax1.axhline(1.0, color='gray', ls='--', lw=1.5)

    if characteristic_tau_a is not None:
        ax1.axvline(
            characteristic_tau_a,
            color=color_a,
            ls=':',
            lw=2
        )

    ax1.set_xscale('log')
    ax1.set_xlabel('Lag')
    ax1.set_ylabel('SF')
    ax1.set_title(f'{label_a} Structure Function')
    ax1.legend(framealpha=0.85)
    ax1.minorticks_on()

    # Panel 2
    ax2 = fig.add_subplot(gs[0, 1])

    ax2.errorbar(
        tau_b,
        sf_b,
        yerr=sf_b_err,
        fmt='s',
        color=color_b,
        ms=6,
        elinewidth=1.2,
        capsize=3,
        label=label_b
    )

    ax2.axhline(1.0, color='gray', ls='--', lw=1.5)

    if characteristic_tau_b is not None:
        ax2.axvline(
            characteristic_tau_b,
            color=color_b,
            ls=':',
            lw=2
        )

    ax2.set_xscale('log')
    ax2.set_xlabel('Lag')
    ax2.set_ylabel('SF')
    ax2.set_title(f'{label_b} Structure Function')
    ax2.legend(framealpha=0.85)
    ax2.minorticks_on()

    # Panel 3
    ax3 = fig.add_subplot(gs[1, 0])

    ax3.errorbar(
        tau_a,
        sf_a,
        yerr=sf_a_err,
        fmt='o',
        color=color_a,
        ms=6,
        capsize=3,
        label=label_a
    )

    ax3.errorbar(
        tau_b,
        sf_b,
        yerr=sf_b_err,
        fmt='s',
        color=color_b,
        ms=6,
        capsize=3,
        label=label_b
    )

    ax3.axhline(1.0, color='gray', ls='--', lw=1.5)

    ax3.set_xscale('log')
    ax3.set_xlabel('Lag')
    ax3.set_ylabel('SF')
    ax3.set_title('Structure Function Comparison')
    ax3.legend(framealpha=0.85)
    ax3.minorticks_on()

    # Panel 4
    ax4 = fig.add_subplot(gs[1, 1])

    ax4.errorbar(
        lag_dcf,
        dcf_vals,
        yerr=dcf_err,
        fmt='D',
        color='purple',
        ms=6,
        elinewidth=1.2,
        capsize=3,
        label='DCF'
    )

    if p95 is not None:
        p95_lower, p95_upper = p95
        ax4.fill_between(
            lag_dcf,
            p95_lower,
            p95_upper,
            color='gray',
            alpha=0.25,
            label='95% \nBootstrap \nEnvelope'
        )

    if p99 is not None:
        p99_lower, p99_upper = p99
        ax4.fill_between(
            lag_dcf,
            p99_lower,
            p99_upper,
            color='gray',
            alpha=0.15,
            label='99% \nBootstrap \nEnvelope'
        )

    ax4.axhline(0.0, color='gray', ls='--', lw=1.5)
    ax4.axvline(0.0, color='gray', ls=':', lw=1.0)

    if peak_lag is not None:
        ax4.axvline(
            peak_lag,
            color='purple',
            ls=':',
            lw=2
        )

    ax4.set_xlabel('Cross Lag')
    ax4.set_ylabel('DCF')
    ax4.set_title('DCF with Bootstrap Significance')
    ax4.legend(
    loc='lower left',
    bbox_to_anchor=(0.75, 0.05),
    ncol=1,
    framealpha=0.9)
    
    ax4.minorticks_on()
    
    fig.tight_layout()
    return fig
