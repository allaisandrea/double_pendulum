"""Draws the figures of world_model_experiments.md from the CSV files in
docs/results, into docs/figures:

    uv run --group docs python docs/figures.py

docs/results.py regenerates the CSVs from their sources (W&B,
world_model.profile, imagination.sim2real, imagination.ranking).
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt

DOCS = Path(__file__).parent
RESULTS, FIGURES = DOCS / "results", DOCS / "figures"

plt.rcParams.update({
    "figure.facecolor": "white",  # readable on dark themes too
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
    "legend.frameon": False,
    "svg.fonttype": "none",  # text stays text: smaller, searchable
})
BLUE, ORANGE, GREEN, RED, PURPLE, GREY = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#7f7f7f"

TRAINED_COLOR = {"wm3": BLUE, "rollout3-k8": ORANGE}
TRAINED_LABEL = {"wm3": "trained in wm3", "rollout3-k8": "trained in K = 8", None: "trained in older models"}
# The ranked world models, in the order the figures show them.
MODELS = {
    "wm3": "wm3 (256 × 3)", "wm-w512-d3-cd50k": "512 × 3, phase 1", "rollout-k1": "K = 1",
    "rollout-k4": "K = 4", "rollout3-k6": "K = 6", "rollout3-k8": "K = 8",
    "rollout3-k8-seed1": "K = 8, seed 1", "rollout3-k10": "K = 10", "rollout3-k12": "K = 12",
    "rollout-k16": "K = 16", "rollout2-k32": "K = 32", "rollout3-k64-clip": "K = 64",
}
# The rollout runs that make up the K series: one per K, 512 × 3, 50k
# steps, lr 2e-3 (K = 64 with gradient clipping and lr 1e-3, as lr 2e-3
# diverged).
K_SERIES = ["rollout-k1", "rollout-k4", "rollout3-k6", "rollout3-k8", "rollout3-k10", "rollout3-k12",
            "rollout-k16", "rollout2-k32", "rollout3-k64-clip"]


def k_axis(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 4, 8, 16, 32, 64], ["1", "4", "8", "16", "32", "64"])
    ax.minorticks_off()
    ax.set_xlabel("K, frames per training rollout")


def read(name: str) -> list[dict]:
    with open(RESULTS / f"{name}.csv", newline="") as f:
        return list(csv.DictReader(f))


def runs(name: str) -> dict[str, dict]:
    """world_model.report's rows by run, the finished ones (a run resumed
    after a crash appears once per attempt)."""
    return {r["run"]: r for r in read(name) if r["state"] == "finished"}


def num(x: str) -> float:
    return float(x) if x else float("nan")


def k_of(run: dict) -> int:
    return int(num(run["rollout_train"])) if run["rollout_train"] else 1


def save(fig, name: str, rect=None):
    fig.tight_layout(rect=rect)
    fig.savefig(FIGURES / f"{name}.svg")
    plt.close(fig)


def profiling():
    rows = read("profile_mps") + read("profile_cloud")
    ms = {(r["device"], r["variant"], int(r["width"]), int(r["depth"]), int(r["batch"])): num(r["ms_per_step"])
          for r in rows}
    shown = [(256, 3, 1024), (256, 3, 16384), (1024, 3, 4096), (1024, 5, 4096)]
    params = {(int(r["width"]), int(r["depth"])): int(r["parameters"]) for r in rows}
    devices = [("mps", "plain", "Mac (M4)", GREY), ("Tesla T4", "plain", "T4", PURPLE),
               ("NVIDIA L4", "plain", "L4", BLUE), ("NVIDIA L4", "compile+bf16", "L4, compile + bf16", GREEN)]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    height = 0.8 / len(devices)
    for d, (device, variant, label, color) in enumerate(devices):
        y = [i + (d - 1.5) * height for i in range(len(shown))]
        ax.barh(y, [ms.get((device, variant, *s), float("nan")) for s in shown], height, label=label, color=color)
    ax.set_yticks(range(len(shown)),
                  [f"{w} × {d} ({params[w, d] / 1e6:.2f}M)\nbatch {b:,}" for w, d, b in shown])
    ax.invert_yaxis()
    ax.set_xlabel("ms per training step")
    ax.legend(loc="upper right")
    ax.grid(axis="y", visible=False)
    save(fig, "profiling")


def batch():
    rows = sorted(runs("batch").values(), key=lambda r: (int(r["batch_size"]), num(r["lr"]), r["bf16"]))
    labels = [f"batch {int(r['batch_size']):,}, lr {num(r['lr']):g}, {'bf16' if r['bf16'] == 'True' else 'fp32'}"
              for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.6), sharey=True)
    for ax, key, title in zip(axes, ("h016", "h064"), ("16 frames", "64 frames")):
        ax.plot([num(r[f"val_policy_wm3/{key}"]) for r in rows], range(len(rows)), "o", color=BLUE, markersize=7)
        ax.set_title(f"1 − R², {title}")
    axes[0].set_yticks(range(len(rows)), labels)
    axes[0].set_ylim(len(rows) - 0.5, -0.5)
    save(fig, "batch")


def scaling():
    # world_model.scale's cooldown branches, by the windows each trained on:
    # its steps, cooldown included, times its batch.
    rows = [r for r in runs("scaling").values() if r["branch_step"]]
    windows = lambda r: int(r["step"]) * int(r["batch_size"]) / 1e6
    colors = {256: BLUE, 512: ORANGE, 1024: GREEN, 2048: RED}
    fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharey=True)
    best = min(rows, key=lambda r: num(r["val_policy_wm3/h016"]))
    for ax, depth in zip(axes, (3, 5)):
        for width, color in colors.items():
            series = sorted((windows(r), num(r["val_policy_wm3/h016"]))
                            for r in rows if int(r["hidden"]) == width and int(r["layers"]) == depth)
            ax.plot(*zip(*series), "o-", color=color, label=f"width {width}")
        ax.set_xscale("log")
        ticks = sorted({x for x, _ in series})
        ax.set_xticks(ticks, [f"{x:.0f}M" for x in ticks])
        ax.minorticks_off()
        ax.set_xlabel("training windows")
        ax.set_title(f"depth {depth}")
        if int(best["layers"]) == depth:
            ax.plot([windows(best)], [num(best["val_policy_wm3/h016"])],
                    "*", color="black", markersize=12, label=f"best, {best['hidden']} × {best['layers']}")
    axes[0].set_ylabel("1 − R², 16 frames, val_policy_wm3")
    axes[0].legend(fontsize=8)
    save(fig, "scaling")


def rollout():
    by = runs("rollout")
    rows = [by[r] for r in K_SERIES]
    k = [k_of(r) for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
    ax = axes[0]
    ax.plot(k, [num(r["val_policy_wm3/h016"]) for r in rows], "o-", color=BLUE, label="val_policy_wm3")
    ax.plot(k, [num(r["val_random_walk/h016"]) for r in rows], "o--", color=GREY, label="val_random_walk")
    ax.set_title("16 frames ahead")
    ax.set_ylabel("1 − R²")
    ax.legend(fontsize=8)
    ax = axes[1]
    for key, label, color, style in [("h064", "64 frames", BLUE, "o-"), ("h125", "125 frames", GREEN, "o-"),
                                     ("upright/h064", "from upright, 64", BLUE, "s--"),
                                     ("upright/h125", "from upright, 125", GREEN, "s--")]:
        ax.plot(k, [num(r[f"val_policy_wm3/{key}"]) for r in rows], style, color=color, label=label)
    ax.axhline(1, color=RED, lw=1, ls=":")
    ax.text(1, 1.02, "copying the last frame", color=RED, fontsize=8, va="bottom")
    ax.set_title("0.5 s and 1 s ahead")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1, 0.5))
    for ax in axes:
        k_axis(ax)
    save(fig, "rollout")


def sim2real():
    rows = read("sim2real")
    real = {r["policy"]: num(r["real"]) for r in rows}
    models = [m for m in K_SERIES if m in rows[0]]
    k = [k_of(runs("rollout")[m]) for m in models]
    ranked = {r["policy"]: num(r["rig"]) for r in read("ranking_scores")}
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3))
    ax = axes[0]
    wm3 = next(r for r in rows if r["policy"] == "ppo-wm3")
    ax.plot(k, [num(wm3[m]) for m in models], "o-", color=BLUE, label="predicted")
    ax.axhline(real["ppo-wm3"], color=RED, ls="--", lw=1, label="rig, from hanging")
    ax.axhline(ranked["ppo-wm3"], color=RED, ls=":", lw=1, label="rig, ranking drives")
    ax.set_title("ppo-wm3, sum of cosines")
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot(k, [sum(abs(num(r[m]) - real[r["policy"]]) for r in rows) / len(rows) for m in models], "o-",
            color=GREY)
    ax.set_title(f"mean absolute error, {len(rows)} policies")
    for ax in axes:
        k_axis(ax)
    save(fig, "sim2real")


def trained(row: dict) -> str | None:
    return row["trained_in"] if row["trained_in"] in TRAINED_COLOR else None


def ranking_rig():
    rows = sorted(read("ranking_scores"), key=lambda r: num(r["rig"]))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for i, r in enumerate(rows):
        ax.barh(i, num(r["rig"]), xerr=num(r["rig_sem"]), color=TRAINED_COLOR.get(trained(r), GREY), capsize=2)
    ax.set_yticks(range(len(rows)), [r["policy"] for r in rows])
    episodes = sorted({r["episodes"] for r in rows})
    ax.set_xlabel(f"sum of cosines per frame on the rig (± sem, {'/'.join(episodes)} drives)")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=TRAINED_COLOR.get(t, GREY), label=label)
                       for t, label in TRAINED_LABEL.items()], loc="lower right", fontsize=8)
    save(fig, "ranking_rig")


def ranking_scatter():
    rows = read("ranking_scores")
    shown = ["wm3", "rollout-k1", "rollout3-k6", "rollout3-k8", "rollout-k16", "rollout3-k64-clip"]
    agreement = {r["world_model"]: r for r in read("ranking_agreement") if r["policies"] == "all"}
    fig, axes = plt.subplots(2, 3, figsize=(8, 5.6), sharex=True, sharey=True)
    top = max(num(r[m]) for r in rows for m in shown) + 0.1
    for ax, model in zip(axes.flat, shown):
        ax.plot([0, top], [0, top], color=GREY, lw=1, ls=":")
        for r in rows:
            own = r["trained_in"] == model
            ax.errorbar(num(r["rig"]), num(r[model]), xerr=num(r["rig_sem"]),
                        fmt="*" if own else "o", markersize=11 if own else 5,
                        color=TRAINED_COLOR.get(trained(r), GREY), mec="black" if own else "none",
                        mew=0.6, elinewidth=0.8)
        a = agreement[model]
        ax.set_title(f"{MODELS[model]}\nMMRV {num(a['mmrv']):.2f}, Spearman {num(a['spearman']):.2f}",
                     fontsize=9)
        ax.set_xlim(0, max(num(r["rig"]) for r in rows) + 0.25)
        ax.set_ylim(0, top)
    for ax in axes[1]:
        ax.set_xlabel("rig")
    for ax in axes[:, 0]:
        ax.set_ylabel("world model")
    handles = [plt.Line2D([], [], marker="o", ls="", color=TRAINED_COLOR.get(t, GREY), label=label)
               for t, label in TRAINED_LABEL.items()]
    handles.append(plt.Line2D([], [], marker="*", ls="", markersize=11, color="white", mec="black",
                              label="trained in this world model"))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8)
    save(fig, "ranking_scatter", rect=(0, 0.05, 1, 1))


def ranking_agreement():
    by = {(r["world_model"], r["policies"]): r for r in read("ranking_agreement")}
    models = [m for m in MODELS if (m, "all") in by]
    subsets = [("all", BLUE), ("new", ORANGE), ("held_out", GREEN)]
    labels = {"all": f"all {by[models[0], 'all']['n']} policies",
              "new": f"{by[models[0], 'new']['n']} new to the rig",
              "held_out": "held out: not trained in the model (where some were)"}
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 4), sharey=True)
    height = 0.8 / len(subsets)
    for ax, key, title in zip(axes, ("mmrv", "spearman", "mae"),
                              ("MMRV (lower is better)", "Spearman (higher is better)",
                               "mean absolute error")):
        for j, (subset, color) in enumerate(subsets):
            y = [i + (j - 1) * height for i in range(len(models))]
            x = [num(by[m, subset][key]) if (m, subset) in by else float("nan") for m in models]
            ax.barh(y, x, height, color=color, label=labels[subset])
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(models)), [MODELS[m] for m in models])
    axes[0].invert_yaxis()
    axes[1].set_xlim(0, 1)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=3, fontsize=8)
    save(fig, "ranking_agreement", rect=(0, 0.06, 1, 1))


def ranking_exploitation():
    """Each policy's overrating in each world model, its own policies starred."""
    rows = read("ranking_scores")
    models = [m for m in MODELS if m in rows[0]]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.axhline(0, color=GREY, lw=1)
    for i, m in enumerate(models):
        for j, r in enumerate(rows):
            own = r["trained_in"] == m
            jitter = (j / (len(rows) - 1) - 0.5) * 0.5
            ax.plot(i + jitter, num(r[m]) - num(r["rig"]), "*" if own else "o",
                    markersize=10 if own else 4, color=TRAINED_COLOR.get(trained(r), GREY),
                    mec="black" if own else "none", mew=0.6)
    ax.set_xticks(range(len(models)), [MODELS[m] for m in models], rotation=35, ha="right")
    ax.set_ylabel("world model − rig")
    ax.grid(axis="x", visible=False)
    handles = [plt.Line2D([], [], marker="o", ls="", color=TRAINED_COLOR.get(t, GREY), label=label)
               for t, label in TRAINED_LABEL.items()]
    handles.append(plt.Line2D([], [], marker="*", ls="", markersize=10, color="white", mec="black",
                              label="trained in this world model"))
    ax.legend(handles=handles, fontsize=8, loc="upper right")
    save(fig, "ranking_exploitation")


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    for draw in (profiling, batch, scaling, rollout, sim2real, ranking_rig, ranking_scatter,
                 ranking_agreement, ranking_exploitation):
        draw()
    print(f"wrote {len(list(FIGURES.glob('*.svg')))} figures to {FIGURES}")
