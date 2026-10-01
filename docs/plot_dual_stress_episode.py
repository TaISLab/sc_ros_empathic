#!/usr/bin/env python3
"""
plot_dual_stress_episode.py
----------------------------
Method/results figure: eta_h / eta_r / eta_s and the two safety-relevant
observables (human joint margin m_min, robot manipulability w_qr) over
one E_extended_m4 trial, with the "compound stress" windows highlighted
-- cycles where the human is inside its joint-limit caution band
(m_min < proximity_threshold) AND the robot is simultaneously in its
worst manipulability quartile for that trial. These are the moments
that motivate having both the joint_safety and manipulability factors
active together: the controller visibly derates eta_h, eta_r AND eta_s
at once, not just the human's term.

Usage:
    python3 docs/plot_dual_stress_episode.py \
        dataset/S03/S03_E_extended_m4_2026-09-11_12-19-16.csv \
        docs/fig_dual_stress_episode

Writes <out>.pdf and <out>.png.
"""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def find_segments(mask_index, gap=5):
    """Group a sorted index of True rows into contiguous (start, end) pairs,
    tolerating gaps of up to `gap` rows (brief single-cycle dropouts)."""
    segs = []
    if len(mask_index) == 0:
        return segs
    start = prev = mask_index[0]
    for i in mask_index[1:]:
        if i - prev > gap:
            segs.append((start, prev))
            start = i
        prev = i
    segs.append((start, prev))
    return segs


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else \
        "dataset/S03/S03_E_extended_m4_2026-09-11_12-19-16.csv"
    out_base = sys.argv[2] if len(sys.argv) > 2 else \
        "docs/fig_dual_stress_episode"

    sidecar = csv_path[:-4] + ".params.json"
    tau = 0.3
    if os.path.exists(sidecar):
        tau = json.load(open(sidecar))["factors"]["proximity_threshold"]

    df = pd.read_csv(csv_path)
    laps = df[df["lap"] >= 1].reset_index(drop=True)
    t = laps["t_rel"].values
    t0 = t[0]
    t = t - t0

    wqr_q1 = laps["w_qr"].quantile(0.25)
    compound = (laps["m_min"] < tau) & (laps["w_qr"] < wqr_q1)
    segs = find_segments(laps.index[compound].to_numpy())

    subj, cond = os.path.basename(csv_path).split("_")[0], "E_extended_m4"

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 7.0), sharex=True,
                              gridspec_kw=dict(height_ratios=[1.2, 1, 1]))

    def shade(ax):
        for s, e in segs:
            ax.axvspan(t[s], t[e], color="#d62728", alpha=0.12, lw=0,
                       zorder=0)
        for lap_n in range(int(laps["lap"].min()), int(laps["lap"].max()) + 1):
            lap_start = t[laps["lap"].values == lap_n]
            if len(lap_start):
                ax.axvline(lap_start[0], color="0.85", lw=0.7, zorder=0)

    ax = axes[0]
    ax.plot(t, laps["eta_h"], color="#1f77b4", lw=1.1, label=r"$\eta_h$")
    ax.plot(t, laps["eta_r"], color="#2ca02c", lw=1.1, label=r"$\eta_r$")
    ax.plot(t, laps["eta_s"], color="#7f0e8f", lw=1.4, label=r"$\eta_s$")
    ax.set_ylabel("efficiency " + r"$\eta$")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", ncol=3, frameon=False, fontsize=9)
    shade(ax)

    ax = axes[1]
    ax.plot(t, laps["m_min"], color="#d62728", lw=1.2, label=r"$m_{\min}$ (human)")
    ax.axhline(tau, color="#d62728", lw=0.9, ls="--",
               label=r"caution band $\tau$=%.2f" % tau)
    ax.set_ylabel(r"joint margin $m_{\min}$")
    ax.set_ylim(0, max(0.9, laps["m_min"].max() * 1.1))
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    shade(ax)

    ax = axes[2]
    ax.plot(t, laps["w_qr"], color="#ff7f0e", lw=1.2, label=r"$w(q_r)$ (robot)")
    ax.axhline(wqr_q1, color="#ff7f0e", lw=0.9, ls="--",
               label="worst quartile (this trial)")
    ax.set_ylabel(r"manipulability $w(q_r)$")
    ax.set_xlabel("time (s), training lap discarded")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    shade(ax)

    fig.suptitle(
        "%s, E_extended_m4, stressed placement -- compound human+robot "
        "stress (shaded, %.1f%% of trial)" % (subj, 100 * compound.mean()),
        fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    fig.savefig(out_base + ".pdf")
    fig.savefig(out_base + ".png", dpi=200)
    print("wrote", out_base + ".pdf", "and", out_base + ".png")
    print("compound-stress fraction: %.3f" % compound.mean())
    print("segments (t_rel s):", [(round(t[s], 1), round(t[e], 1)) for s, e in segs])


if __name__ == "__main__":
    main()
