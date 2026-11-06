import argparse
import csv
import glob
import os
import pickle
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle


from rescue_mission import RescueEnv

ALGO_LABEL = {"qlearning": "Q-learning", "mc": "Monte Carlo"}
ALGO_COLOR = {"qlearning": "tab:blue", "mc": "tab:orange"}
MODE_LABEL = {"A": "Mode A (cứu gần trước)", "B": "Mode B (ưu tiên giá trị cao)"}


def load_results(d):
    res = []
    for f in sorted(glob.glob(os.path.join(d, "*.pkl"))):
        with open(f, "rb") as fh:
            res.append(pickle.load(fh))
    return res


def smooth(x, w):
    x = np.asarray(x, dtype=np.float64)
    if w <= 1 or len(x) < w:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    ma = (c[w:] - c[:-w]) / w
    return np.concatenate([np.full(w - 1, np.nan), ma])


def select(res, **kw):
    return [r for r in res if all(r[k] == v for k, v in kw.items())]


# ----------------------------------------------------------------------
# 1) Learning curve
# ----------------------------------------------------------------------
def plot_learning(res, map_name, variant, out_dir):
    modes = sorted({r["mode"] for r in res if r["map"] == map_name and r["variant"] == variant})
    if not modes:
        return
    fig, axes = plt.subplots(2, len(modes), figsize=(6.2 * len(modes), 7), squeeze=False)
    for j, mode in enumerate(modes):
        for algo in ALGO_LABEL:
            runs = select(res, map=map_name, variant=variant, mode=mode, algo=algo)
            if not runs:
                continue
            for i, key in enumerate(("returns", "rescued")):
                data = np.stack([r[key] for r in runs]).astype(np.float64)
                w = max(1, data.shape[1] // 50)
                sm = np.stack([smooth(row, w) for row in data])
                m, s = np.nanmean(sm, 0), np.nanstd(sm, 0)
                x = np.arange(1, data.shape[1] + 1)
                ax = axes[i, j]
                ax.plot(x, m, color=ALGO_COLOR[algo],
                        label=f"{ALGO_LABEL[algo]} ({len(runs)} seed)")
                ax.fill_between(x, m - s, m + s, color=ALGO_COLOR[algo], alpha=0.2)
        N = RescueEnv(map_name).N
        axes[0, j].set_title(f"{map_name} – {MODE_LABEL[mode]}")
        axes[0, j].set_ylabel("Return / episode")
        axes[1, j].set_ylabel(f"Số nạn nhân cứu (tối đa {N})")
        axes[1, j].set_ylim(-0.1, N + 0.2)
        for i in range(2):
            axes[i, j].set_xlabel("Episode")
            axes[i, j].grid(alpha=0.3)
            axes[i, j].legend(loc="lower right")
    fig.suptitle(f"Learning curve – {map_name} (reward: {variant}, làm mượt, mean ± std)", y=1.0)
    fig.tight_layout()
    path = os.path.join(out_dir, f"learning_{map_name}_{variant}.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


# ----------------------------------------------------------------------
# 2) Trực quan hóa policy
# ----------------------------------------------------------------------
def draw_policy(ax, env, ro, title):
    H, W = env.H, env.W
    for r in range(H):
        for c in range(W):
            ch = env.layout[r][c]
            color = {"#": "#444444", "D": "#f4a6a6", "G": "#a8e0a8",
                     "M": "#a9c9f5", "S": "#fff3a8"}.get(ch, "white")
            ax.add_patch(Rectangle((c, r), 1, 1, facecolor=color, edgecolor="#bbbbbb"))
            if ch in "SGMD":
                ax.text(c + 0.12, r + 0.2, {"D": "⚠"}.get(ch, ch), fontsize=9,
                        va="center", color="#333333")
    rank = {v: k + 1 for k, v in enumerate(ro["order"])}
    for (r, c), i in env.victim_pos.items():
        v = env.victims[i]
        face = "#ffb347" if i in rank else "#dddddd"
        ax.add_patch(Circle((c + 0.5, r + 0.5), 0.3, facecolor=face, edgecolor="k"))
        label = f"V{i}" + ("+" if v["needs_medkit"] else "")
        ax.text(c + 0.5, r + 0.42, label, ha="center", va="center", fontsize=8, weight="bold")
        ax.text(c + 0.5, r + 0.66, f"v={v['value']}", ha="center", va="center", fontsize=6.5)
        if i in rank:
            ax.text(c + 0.88, r + 0.14, f"#{rank[i]}", ha="center", va="center", fontsize=9,
                    color="darkred", weight="bold")
    path = ro["path"]
    n = max(1, len(path) - 1)
    cmap = plt.get_cmap("viridis")
    for k in range(len(path) - 1):
        (r0, c0), (r1, c1) = path[k], path[k + 1]
        if (r0, c0) == (r1, c1):
            continue
        off = 0.08 * ((k % 3) - 1)          # lệch nhẹ để thấy các lần đi lại cùng đoạn
        ax.annotate("", xy=(c1 + 0.5 + off, r1 + 0.5 + off), xytext=(c0 + 0.5 + off, r0 + 0.5 + off),
                    arrowprops=dict(arrowstyle="-|>", color=cmap(k / n), lw=1.3, alpha=0.85))
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9)


def _best_run(runs):
    return max(runs, key=lambda r: (r["final"]["mean_rescued"], r["final"]["mean_return"]))


def plot_policy(res, map_name, variant, out_dir, algo="qlearning"):
    modes = sorted({r["mode"] for r in res if r["map"] == map_name and r["variant"] == variant})
    cols = [(m, select(res, map=map_name, variant=variant, mode=m, algo=algo)) for m in modes]
    cols = [(m, runs) for m, runs in cols if runs]
    if not cols:
        return
    fig, axes = plt.subplots(1, len(cols), figsize=(5.2 * len(cols), 5.4), squeeze=False)
    for ax, (mode, runs) in zip(axes[0], cols):
        r = _best_run(runs)
        env = RescueEnv(map_name, reward_mode=mode)
        ro = r["rollout"]
        order = " → ".join(f"V{i}" for i in ro["order"]) or "không cứu được ai"
        draw_policy(ax, env, ro, f"{MODE_LABEL[mode]}\nThứ tự cứu: {order}\n"
                                 f"{ro['steps']} bước, return {ro['total_return']:.0f}")
    fig.suptitle(f"Policy cuối ({ALGO_LABEL[algo]}) – {map_name}, reward: {variant}", y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, f"policy_{map_name}_{variant}.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_policy_variants(res, map_name, out_dir, mode="A", algo="qlearning"):
    variants = sorted({r["variant"] for r in res if r["map"] == map_name and r["mode"] == mode})
    cols = [(v, select(res, map=map_name, variant=v, mode=mode, algo=algo)) for v in variants]
    cols = [(v, runs) for v, runs in cols if runs]
    if len(cols) < 2:
        return
    fig, axes = plt.subplots(1, len(cols), figsize=(5.2 * len(cols), 5.4), squeeze=False)
    for ax, (v, runs) in zip(axes[0], cols):
        r = _best_run(runs)
        env = RescueEnv(map_name, reward_mode=mode, reward_cfg=None)
        ro = r["rollout"]
        order = " → ".join(f"V{i}" for i in ro["order"]) or "không cứu được ai"
        draw_policy(ax, env, ro, f"reward: {v}\nThứ tự cứu: {order}\n{ro['steps']} bước")
    fig.suptitle(f"Ảnh hưởng của cấu hình reward – {map_name}, {MODE_LABEL[mode]}", y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, f"policy_variants_{map_name}.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


# ----------------------------------------------------------------------
# 3) Bảng thống kê
# ----------------------------------------------------------------------
def summary_table(res, out_dir):
    groups = defaultdict(list)
    for r in res:
        groups[(r["map"], r["variant"], r["mode"], r["algo"])].append(r)
    rows = []
    for (m, var, mode, algo), runs in sorted(groups.items()):
        fin = lambda k: np.array([r["final"][k] for r in runs], dtype=float)
        last = np.array([np.mean(r["returns"][-max(50, len(r["returns"]) // 20):]) for r in runs])
        rows.append(dict(
            map=m, reward=var, mode=mode, algo=ALGO_LABEL[algo], seeds=len(runs),
            episodes=runs[0]["episodes"], n_states=runs[0]["n_states"],
            greedy_return=f"{fin('mean_return').mean():.1f} ± {fin('mean_return').std():.1f}",
            success_rate=f"{fin('success_rate').mean() * 100:.0f}%",
            rescued=f"{fin('mean_rescued').mean():.2f}",
            steps=f"{fin('mean_steps').mean():.1f}",
            train_return_last5pct=f"{last.mean():.1f} ± {last.std():.1f}",
            train_time_s=f"{np.mean([r['train_time'] for r in runs]):.0f}",
        ))
    path = os.path.join(out_dir, "summary.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    # in markdown
    keys = list(rows[0].keys())
    print("| " + " | ".join(keys) + " |")
    print("|" + "---|" * len(keys))
    for r in rows:
        print("| " + " | ".join(str(r[k]) for k in keys) + " |")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results")
    args = ap.parse_args()
    res = load_results(args.dir)
    if not res:
        raise SystemExit(f"Không có file .pkl trong {args.dir}/ — chạy train.py trước.")
    fig_dir = os.path.join(args.dir, "figs")
    os.makedirs(fig_dir, exist_ok=True)

    maps = sorted({r["map"] for r in res})
    variants = sorted({r["variant"] for r in res})
    made = []
    for m in maps:
        for v in variants:
            made += [plot_learning(res, m, v, fig_dir), plot_policy(res, m, v, fig_dir)]
        made.append(plot_policy_variants(res, m, fig_dir))
    print("\n== Bảng thống kê ==")
    made.append(summary_table(res, args.dir))
    print("\nĐã tạo:")
    for p in made:
        if p:
            print(" ", p)


if __name__ == "__main__":
    main()
