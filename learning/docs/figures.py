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

TRAINED_COLOR = {"wm3": BLUE, "rollout3-k8": ORANGE, "rollout-k1": GREEN, "rollout3-k6": PURPLE, "stoch-b05": RED}
TRAINED_LABEL = {"wm3": "trained in wm3", "rollout3-k8": "trained in K = 8", "rollout-k1": "trained in K = 1",
                 "rollout3-k6": "trained in K = 6", "stoch-b05": "trained in the stochastic model",
                 None: "trained in older models"}
# The ranked world models, in the order the figures show them.
MODELS = {
    "wm3": "wm3 (256 × 3)", "wm-w512-d3-cd50k": "512 × 3, phase 1", "rollout-k1": "K = 1",
    "rollout-k4": "K = 4", "rollout3-k6": "K = 6", "rollout3-k8": "K = 8",
    "rollout3-k8-seed1": "K = 8, seed 1", "rollout3-k10": "K = 10", "rollout3-k12": "K = 12",
    "rollout-k16": "K = 16", "rollout2-k32": "K = 32", "rollout3-k64-clip": "K = 64",
    "stoch-b05@0": "stochastic, mean", "stoch-b05": "stochastic, sampled", "stoch-b0": "stochastic β = 0, sampled",
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
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for i, r in enumerate(rows):
        ax.barh(i, num(r["rig"]), xerr=num(r["rig_sem"]), color=TRAINED_COLOR.get(trained(r), GREY), capsize=2)
    ax.set_yticks(range(len(rows)), [r["policy"] for r in rows])
    episodes = sorted({r["episodes"] for r in rows})
    ax.set_xlabel(f"sum of cosines per frame on the rig (± sem, {'/'.join(episodes)} drives)")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=TRAINED_COLOR.get(t, GREY), label=label)
                       for t, label in TRAINED_LABEL.items()], loc="lower right", fontsize=8)
    save(fig, "ranking_rig")


# The stochastic model and the deterministic ones it is compared with in its section.
COMPARED = ["rollout-k1", "rollout3-k6", "stoch-b05@0", "stoch-b05"]


def own(row: dict, model: str) -> bool:
    """Whether the policy was trained in `model` (a stochastic model's
    columns at another tau, MODEL@TAU, count as the model)."""
    return row["trained_in"] == model.split("@")[0]


def ranking_scatter(shown=("wm3", "rollout-k1", "rollout3-k6", "rollout3-k8", "stoch-b05@0", "stoch-b05"),
                    name="ranking_scatter", cols=3):
    rows = read("ranking_scores")
    agreement = {r["world_model"]: r for r in read("ranking_agreement") if r["policies"] == "all"}
    lines = -(-len(shown) // cols)
    fig, axes = plt.subplots(lines, cols, figsize=(2.7 * cols, 2.6 * lines + 0.8), sharex=True, sharey=True,
                             squeeze=False)
    top = max(num(r[m]) for r in rows for m in shown) + 0.1
    for ax, model in zip(axes.flat, shown):
        ax.plot([0, top], [0, top], color=GREY, lw=1, ls=":")
        for r in rows:
            own_ = own(r, model)
            ax.errorbar(num(r["rig"]), num(r[model]), xerr=num(r["rig_sem"]),
                        fmt="*" if own_ else "o", markersize=11 if own_ else 5,
                        color=TRAINED_COLOR.get(trained(r), GREY), mec="black" if own_ else "none",
                        mew=0.6, elinewidth=0.8)
        a = agreement[model]
        ax.set_title(f"{MODELS[model]}\nMMRV {num(a['mmrv']):.2f}, Spearman {num(a['spearman']):.2f}",
                     fontsize=9)
        ax.set_xlim(0, max(num(r["rig"]) for r in rows) + 0.25)
        ax.set_ylim(0, top)
    for ax in axes[-1]:
        ax.set_xlabel("rig")
    for ax in axes[:, 0]:
        ax.set_ylabel("world model")
    handles = [plt.Line2D([], [], marker="o", ls="", color=TRAINED_COLOR.get(t, GREY), label=label)
               for t, label in TRAINED_LABEL.items()]
    handles.append(plt.Line2D([], [], marker="*", ls="", markersize=11, color="white", mec="black",
                              label="trained in this world model"))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8)
    save(fig, name, rect=(0, 0.1 if lines > 1 else 0.2, 1, 1))


def ranking_agreement():
    by = {(r["world_model"], r["policies"]): r for r in read("ranking_agreement")}
    models = [m for m in MODELS if (m, "all") in by]
    subsets = [("all", BLUE), ("new", ORANGE), ("held_out", GREEN)]
    labels = {"all": f"all {by[models[0], 'all']['n']} policies",
              "new": f"{by[models[0], 'new']['n']} new to the rig",
              "held_out": "held out: not trained in the model (where some were)"}
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 5), sharey=True)
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


def ranking_exploitation(models=None, name="ranking_exploitation", width=9.5):
    """Each policy's overrating in each world model, its own policies starred."""
    rows = read("ranking_scores")
    models = models or [m for m in MODELS if m in rows[0]]
    fig, ax = plt.subplots(figsize=(width, 4.8))
    ax.axhline(0, color=GREY, lw=1)
    for i, m in enumerate(models):
        for j, r in enumerate(rows):
            own_ = own(r, m)
            jitter = (j / (len(rows) - 1) - 0.5) * 0.5
            ax.plot(i + jitter, num(r[m]) - num(r["rig"]), "*" if own_ else "o",
                    markersize=10 if own_ else 4, color=TRAINED_COLOR.get(trained(r), GREY),
                    mec="black" if own_ else "none", mew=0.6)
    ax.set_xticks(range(len(models)), [MODELS[m] for m in models], rotation=35, ha="right")
    ax.set_ylabel("world model − rig")
    ax.grid(axis="x", visible=False)
    handles = [plt.Line2D([], [], marker="o", ls="", color=TRAINED_COLOR.get(t, GREY), label=label)
               for t, label in TRAINED_LABEL.items()]
    handles.append(plt.Line2D([], [], marker="*", ls="", markersize=10, color="white", mec="black",
                              label="trained in this world model"))
    fig.legend(handles=handles, fontsize=8, loc="lower center", ncol=3 if len(models) < 6 else 4)
    save(fig, name, rect=(0, 0.14 if len(models) < 6 else 0.1, 1, 1))


def stochastic_agreement():
    """The stochastic model against K = 1 and K = 6 over the ranked policies:
    ranking (MMRV), accuracy (mean absolute error), and overrating of the
    model's own policies and of the rest."""
    rows = read("ranking_scores")
    by = {r["world_model"]: r for r in read("ranking_agreement") if r["policies"] == "all"}
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
    y = range(len(COMPARED))
    axes[0].barh(y, [num(by[m]["mmrv"]) for m in COMPARED], 0.6, color=BLUE)
    axes[0].set_title("MMRV (lower is better)", fontsize=9)
    axes[1].barh(y, [num(by[m]["mae"]) for m in COMPARED], 0.6, color=BLUE)
    axes[1].set_title("mean absolute error", fontsize=9)
    over = lambda m, keep: sum(num(r[m]) - num(r["rig"]) for r in rows if keep(r)) / sum(1 for r in rows if keep(r))
    axes[2].barh([i - 0.2 for i in y], [over(m, lambda r, m=m: own(r, m)) for m in COMPARED], 0.4, color=RED,
                 label="its own policies")
    axes[2].barh([i + 0.2 for i in y], [over(m, lambda r, m=m: not own(r, m)) for m in COMPARED], 0.4, color=GREY,
                 label="the other policies")
    axes[2].axvline(0, color=GREY, lw=1)
    axes[2].set_title("overrating (world model − rig)", fontsize=9)
    axes[2].legend(fontsize=7, loc="lower right")
    for ax in axes:
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(list(y), [MODELS[m] for m in COMPARED])
    axes[0].invert_yaxis()
    save(fig, "stochastic_agreement")


def stochastic_ensemble():
    """The stochastic model's open-loop accuracy on the balancing validation
    set: its mean rollout and its ensemble against the one-step model."""
    horizons = [1, 4, 16, 64, 125]
    val = "1790464558"
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    for ax, name, title in ((axes[0], "stochastic", "all windows"), (axes[1], "stochastic_upright", "from upright")):
        by = {r["checkpoint"]: r for r in read(name) if r["recording"] == val}
        for key, model, label, color, style in [
                ("one_minus_r2", "rollout-k1", "one-step (K = 1), mean", GREY, "o--"),
                ("one_minus_r2", "stoch-b05", "stochastic, mean rollout", BLUE, "o--"),
                ("ensemble/one_minus_r2", "stoch-b05", "stochastic, mean of 8 samples", BLUE, "o-"),
                ("ensemble/crps", "stoch-b05", "stochastic, CRPS / copy-last", RED, "s-")]:
            ax.plot(horizons, [num(by[model][f"{key}/h{h:03d}"]) for h in horizons], style, color=color, label=label)
        ax.axhline(1, color=RED, lw=1, ls=":")
        ax.set_title(title)
    upright = {r["checkpoint"]: r for r in read("stochastic_upright") if r["recording"] == val}
    alls = {r["checkpoint"]: r for r in read("stochastic") if r["recording"] == val}
    for rows, label, style in ((alls, "all windows", "o-"), (upright, "from upright", "s--")):
        axes[2].plot(horizons, [num(rows["stoch-b05"][f"ensemble/spread_skill/h{h:03d}"]) for h in horizons], style,
                     color=BLUE, label=label)
    axes[2].axhline(1, color=GREY, lw=1, ls=":")
    axes[2].set_ylim(0, 1.2)
    axes[2].set_title("ensemble spread / error")
    axes[0].set_ylabel("1 − R², or CRPS over copying")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(horizons, [str(h) for h in horizons])
        ax.minorticks_off()
        ax.set_xlabel("frames ahead")
    axes[0].legend(fontsize=7)
    axes[2].legend(fontsize=7)
    save(fig, "stochastic_ensemble")


# The loop: each collection's validation recording, with its label, and
# the first world model trained on its collection.
LOOP_VALS = [("1790611634", "collection 0: ppo-stoch + bursts", "stoch-it1"),
             ("1790619211", "collection 1: ppo-stoch-it1", "stoch-it2"),
             ("1790626779", "collection 2: ppo-stoch-it2", "stoch-it3"),
             ("1790464558", "val_policy_wm3 (ppo-wm3)", None)]
LOOP_MODELS = ["stoch-b05", "stoch-it1", "stoch-it2", "stoch-it3"]


def loop_world_models():
    """Each loop world model on each collection's validation recording;
    filled markers where the model trained on that collection."""
    rows = {(r["checkpoint"], r["recording"]): r for r in read("loop_world_models")}
    upright = {(r["checkpoint"], r["recording"]): r for r in read("loop_world_models_upright")}
    colors = [RED, ORANGE, GREEN, GREY]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    panels = [(rows, "nll", "one-step NLL per element"), (rows, "one_minus_r2/h016", "1 − R², 16 frames"),
              (upright, "ensemble/crps/h064", "CRPS / copy-last, 64 frames, from upright")]
    x = range(len(LOOP_MODELS))
    for ax, (table, key, title) in zip(axes, panels):
        for (val, label, first), color in zip(LOOP_VALS, colors):
            ys = [num(table[m, val][key]) for m in LOOP_MODELS]
            ax.plot(x, ys, "-", color=color, label=label)
            trained = [first is not None and LOOP_MODELS.index(m) >= LOOP_MODELS.index(first) for m in LOOP_MODELS]
            for i, (y, t) in enumerate(zip(ys, trained)):
                ax.plot(i, y, "o", color=color, mfc=color if t else "white", markersize=6)
        ax.set_xticks(list(x), ["b05", "it1", "it2", "it3"])
        ax.set_xlabel("world model")
        ax.set_title(title, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], marker="o", ls="", color="black", mfc="black", label="trained on the collection"))
    handles.append(plt.Line2D([], [], marker="o", ls="", color="black", mfc="white", label="not trained on it"))
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8)
    save(fig, "loop_world_models", rect=(0, 0.16, 1, 1))


def loop_rig():
    """Each loop policy's greedy rig score, and each world model's forecast of it."""
    rows = {r["policy"]: r for r in read("loop_scores")}
    policies = [p for p in ["ppo-stoch", "ppo-stoch-it1", "ppo-stoch-it2", "ppo-stoch-it3"] if p in rows]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    width = 0.8 / (1 + len(LOOP_MODELS))
    for i, p in enumerate(policies):
        r = rows[p]
        ax.bar(i - 0.4 + width / 2, num(r["rig"]), width, yerr=num(r["rig_sem"]), color="black", capsize=2,
               label="rig" if i == 0 else None)
        for j, (m, color) in enumerate(zip(LOOP_MODELS, [GREY, BLUE, GREEN, PURPLE])):
            ax.bar(i - 0.4 + width * (j + 1.5), num(r[m]), width, color=color,
                   label=f"{m}, sampled" if i == 0 else None)
    ax.set_xticks(range(len(policies)), policies)
    ax.set_ylim(1.4, None)
    ax.set_ylabel("sum of cosines per frame (greedy)")
    ax.legend(fontsize=7, ncol=3, loc="upper left")
    ax.grid(axis="x", visible=False)
    save(fig, "loop_rig")


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    for draw in (profiling, batch, scaling, rollout, sim2real, ranking_rig, ranking_scatter,
                 ranking_agreement, ranking_exploitation, stochastic_ensemble, stochastic_agreement,
                 lambda: ranking_scatter(COMPARED, "stochastic_scatter", cols=4),
                 lambda: ranking_exploitation(COMPARED, "stochastic_exploitation", width=6),
                 loop_world_models, loop_rig):
        draw()
    print(f"wrote {len(list(FIGURES.glob('*.svg')))} figures to {FIGURES}")
