"""Step 6 - Figures for the paper from results/*.csv."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C

FEATURE_LABEL = {"ecapa": "ECAPA-TDNN", "wavlm": "WavLM", "egemaps": "eGeMAPS"}
NORM_LABEL = {"none": "no normalisation", "rested_baseline": "rested-baseline norm."}


def protocol_comparison(summary, metric="uar", level="utterance"):
    d = summary[(summary.metric == metric) & (summary.level == level)
                & summary.feature.isin(FEATURE_LABEL)]
    groups = [(f, n) for f in FEATURE_LABEL for n in NORM_LABEL]
    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for k, (protocol, colour, label) in enumerate([("loso", "#2b6cb0", "LOSO (speaker-independent)"),
                                                   ("random_split", "#dd6b20", "Random split (speaker-dependent)")]):
        rows = [d[(d.feature == f) & (d.normalisation == n) & (d.protocol == protocol)].iloc[0]
                for f, n in groups]
        v = np.array([r.value for r in rows])
        err = np.array([[r.value - r.ci_low for r in rows], [r.ci_high - r.value for r in rows]])
        ax.bar(x + (k - 0.5) * 0.38, v, 0.38, yerr=err, capsize=3, color=colour, label=label)
    ax.axhline(0.5, color="grey", lw=0.8, ls="--")
    ax.set_xticks(x, [f"{FEATURE_LABEL[f]}\n{NORM_LABEL[n]}" for f, n in groups], fontsize=8)
    ax.set_ylabel(f"{level} {metric.upper()}")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(C.RESULTS / f"fig_protocol_comparison_{metric}.png", dpi=200)
    plt.close(fig)


def per_participant(parts, feature="wavlm", norm="none"):
    """The headline model: WavLM without normalisation (the rested-baseline rows are label-leaky)."""
    p = parts[(parts.feature == feature) & (parts.normalisation == norm) & (parts.protocol == "loso")]
    p = p.sort_values(["segment_accuracy", "utterance_accuracy"])
    colour = p.utterance_accuracy.map({1.0: "#2b6cb0", 0.5: "#dd9b20", 0.0: "#c53030"})
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(np.arange(len(p)), p.segment_accuracy, color=colour)
    ax.axhline(0.5, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("participant (sorted)")
    ax.set_ylabel("segment accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Per-participant LOSO accuracy - {FEATURE_LABEL[feature]}, {NORM_LABEL[norm]}", fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ["#2b6cb0", "#dd9b20", "#c53030"]]
    counts = [(p.utterance_accuracy == v).sum() for v in [1.0, 0.5, 0.0]]
    ax.legend(handles, [f"both recordings correct ({counts[0]})", f"one correct ({counts[1]})",
                        f"neither correct ({counts[2]})"], fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(C.RESULTS / "fig_per_participant.png", dpi=200)
    plt.close(fig)


def main():
    summary = pd.read_csv(C.RESULTS / "metrics_summary.csv")
    parts = pd.read_csv(C.RESULTS / "per_participant_accuracy.csv")
    for metric in ["uar", "roc_auc"]:
        protocol_comparison(summary, metric)
    per_participant(parts)
    print("figures written to", C.RESULTS)


if __name__ == "__main__":
    main()
