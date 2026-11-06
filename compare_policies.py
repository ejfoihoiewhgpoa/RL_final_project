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

from rescue_mission import RescueEnv, ACTION_NAMES

GAMMA = 0.99


def replay(map_name, mode, actions, reward_cfg=None):
    """Phát lại chuỗi action; trả về return (chưa/đã chiết khấu), bước giao từng nạn nhân, bước vào ô D."""
    env = RescueEnv(map_name, reward_mode=mode, reward_cfg=reward_cfg)
    env.reset()
    G, Gd, deliver, haz = 0.0, 0.0, {}, 0
    for t, name in enumerate(actions, start=1):
        before = env.mask
        _, r, te, tr, _ = env.step(ACTION_NAMES.index(name))
        G += r
        Gd += (GAMMA ** (t - 1)) * r
        new = env.mask ^ before
        if new:
            deliver[new.bit_length() - 1] = t
        if env.pos in env.hazards:
            haz += 1
        if te or tr:
            break
    return G, Gd, deliver, haz, env


def metrics(run):
    m, ro = run["map"], run["rollout"]
    env0 = RescueEnv(m)
    values = [v["value"] for v in env0.victims]
    GA, GAd, deliver, haz, _ = replay(m, "A", ro["actions"])
    GB, GBd, _, _, _ = replay(m, "B", ro["actions"])
    ts = np.array([deliver[i] for i in deliver], dtype=float)
    vs = np.array([values[i] for i in deliver], dtype=float)
    vmax = int(np.argmax(values))
    return dict(
        policy=run["mode"], seed=run["seed"], order=ro["order"], steps=ro["steps"],
        saved=len(deliver), n=env0.N,
        ret_A=GA, ret_B=GB, disc_A=GAd, disc_B=GBd,
        avg_t=float(ts.mean()) if len(ts) else np.nan,
        wavg_t=float((vs * ts).sum() / vs.sum()) if len(ts) else np.nan,
        t_top=deliver.get(vmax, np.nan), top_victim=vmax,
        hazard_steps=haz, deliver=deliver, values=values,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results")
    ap.add_argument("--variant", default="base")
    ap.add_argument("--algo", default="qlearning")
    args = ap.parse_args()

    runs = []
    for f in sorted(glob.glob(os.path.join(args.dir, "*.pkl"))):
        with open(f, "rb") as fh:
            r = pickle.load(fh)
        if r["variant"] == args.variant and r["algo"] == args.algo:
            runs.append(r)
    if not runs:
        raise SystemExit("Không có kết quả phù hợp trong results/ — chạy train.py trước.")

    fig_dir = os.path.join(args.dir, "figs")
    os.makedirs(fig_dir, exist_ok=True)
    rows = []
    for m in sorted({r["map"] for r in runs}):
        # lấy seed tốt nhất của mỗi policy (cứu nhiều nhất, return theo reward của chính nó cao nhất)
        best = {}
        for mode in "AB":
            cand = [r for r in runs if r["map"] == m and r["mode"] == mode]
            if cand:
                best[mode] = max(cand, key=lambda r: (r["final"]["mean_rescued"],
                                                      r["final"]["mean_return"]))
        if len(best) < 2:
            continue
        M = {mode: metrics(best[mode]) for mode in "AB"}
        for mode in "AB":
            x = M[mode]
            rows.append(dict(
                map=m, policy=f"Policy {mode}", order=" → ".join(f"V{i}" for i in x["order"]),
                saved=f"{x['saved']}/{x['n']}", steps=x["steps"],
                return_under_rewardA=round(x["ret_A"], 1), return_under_rewardB=round(x["ret_B"], 1),
                disc_return_A=round(x["disc_A"], 1), disc_return_B=round(x["disc_B"], 1),
                avg_delivery_time=round(x["avg_t"], 1),
                value_weighted_delivery_time=round(x["wavg_t"], 1),
                time_to_top_value_victim=x["t_top"], hazard_steps=x["hazard_steps"]))

        # ---- hình ----
        fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
        # (1) return chéo
        w = 0.35
        xs = np.arange(2)
        for k, mode in enumerate("AB"):
            vals = [M[mode]["disc_A"], M[mode]["disc_B"]]
            b = ax[0].bar(xs + (k - 0.5) * w, vals, w, label=f"Policy {mode}",
                          color=("tab:blue" if mode == "A" else "tab:red"))
            for xx, v in zip(xs + (k - 0.5) * w, vals):
                ax[0].text(xx, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
        ax[0].set_xticks(xs); ax[0].set_xticklabels(["đo theo reward A\n(giá trị như nhau)",
                                                     "đo theo reward B\n(nhân theo giá trị)"])
        ax[0].set_title("Return chiết khấu chéo (γ=0,99)"); ax[0].legend()
        ax[0].set_ylabel("Return chiết khấu")
        # (2) timeline giao nạn nhân
        for k, mode in enumerate("AB"):
            x = M[mode]
            for i, t in x["deliver"].items():
                ax[1].scatter(t, k, s=130 * x["values"][i], color=("tab:blue" if mode == "A" else "tab:red"),
                              alpha=0.7, edgecolor="k")
                ax[1].text(t, k + 0.18, f"V{i}", ha="center", fontsize=8)
        ax[1].set_yticks([0, 1]); ax[1].set_yticklabels(["Policy A", "Policy B"])
        ax[1].set_ylim(-0.6, 1.6); ax[1].set_xlabel("Bước giao nạn nhân về G")
        ax[1].set_title("Thời điểm giao nạn nhân (kích thước ∝ giá trị)"); ax[1].grid(alpha=0.3)
        # (3) thời gian giao trung bình
        labels = ["TB thời gian giao\n(không trọng số)", "TB thời gian giao\n(trọng số giá trị)"]
        for k, mode in enumerate("AB"):
            vals = [M[mode]["avg_t"], M[mode]["wavg_t"]]
            ax[2].bar(np.arange(2) + (k - 0.5) * w, vals, w, label=f"Policy {mode}",
                      color=("tab:blue" if mode == "A" else "tab:red"))
            for xx, v in zip(np.arange(2) + (k - 0.5) * w, vals):
                ax[2].text(xx, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        ax[2].set_xticks(np.arange(2)); ax[2].set_xticklabels(labels)
        ax[2].set_title("Càng thấp càng tốt"); ax[2].legend()
        fig.suptitle(f"So sánh hai policy cạnh tranh – {m} ({args.algo}, reward: {args.variant})", y=1.03)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"compare_{m}_{args.variant}.png"), dpi=140, bbox_inches="tight")
        plt.close(fig)

    keys = list(rows[0].keys())
    path = os.path.join(args.dir, f"compare_{args.variant}.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w_ = csv.DictWriter(f, fieldnames=keys); w_.writeheader(); w_.writerows(rows)
    print("| " + " | ".join(keys) + " |"); print("|" + "---|" * len(keys))
    for r in rows:
        print("| " + " | ".join(str(r[k]) for k in keys) + " |")
    print("\nĐã lưu:", path, "và hình compare_*.png trong", fig_dir)


if __name__ == "__main__":
    main()
