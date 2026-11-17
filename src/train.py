import argparse
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from rescue_mission import RescueEnv
from agents import QLearningAgent, MonteCarloAgent, evaluate, rollout

# số episode mặc định cho mỗi map (dùng chung cho cả hai thuật toán để so sánh công bằng)
DEFAULT_EPISODES = {"map1": 20000, "map2": 50000, "map3": 60000}

# các cấu hình reward để so sánh (yêu cầu: thử ít nhất 2 cấu hình)
REWARD_VARIANTS = {
    "base":       {},                          # step -1, hazard -10, rescue 50, bonus 100
    "low_hazard": {"hazard": -2.0},            # ô D ít nguy hiểm -> agent dám đi xuyên
    "high_step":  {"step": -3.0},              # phạt bước nặng -> agent vội hơn
}

ALGOS = {"qlearning": QLearningAgent, "mc": MonteCarloAgent}


def run_job(job):
    map_name, mode, algo, seed, variant, episodes, out_dir, force = job
    fname = os.path.join(out_dir, f"{map_name}_{mode}_{algo}_{variant}_s{seed}.pkl")
    if os.path.exists(fname) and not force:          # đã chạy rồi -> bỏ qua (chạy tiếp khi bị ngắt)
        with open(fname, "rb") as f:
            old = pickle.load(f)
        return fname, old["final"], 0.0
    env = RescueEnv(map_name, reward_mode=mode, reward_cfg=REWARD_VARIANTS[variant])
    agent = ALGOS[algo](env, seed=seed)
    t0 = time.time()
    log = agent.train(episodes)
    train_time = time.time() - t0
    result = dict(
        map=map_name, mode=mode, algo=algo, seed=seed, variant=variant, episodes=episodes,
        returns=np.array(log["returns"], dtype=np.float32),
        rescued=np.array(log["rescued"], dtype=np.int8),
        steps=np.array(log["steps"], dtype=np.int16),
        final=evaluate(env, agent.Q, episodes=1),     # env deterministic -> 1 episode greedy là đủ
        rollout=rollout(env, agent.Q),
        train_time=train_time,
        n_states=env.n_states,
    )
    with open(fname, "wb") as f:
        pickle.dump(result, f)
    return fname, result["final"], train_time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", nargs="+", default=["map1", "map2", "map3"])
    ap.add_argument("--modes", nargs="+", default=["A", "B"])
    ap.add_argument("--algos", nargs="+", default=["qlearning", "mc"], choices=list(ALGOS))
    ap.add_argument("--variants", nargs="+", default=["base"], choices=list(REWARD_VARIANTS))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--scale", type=float, default=1.0, help="nhân số episode mặc định")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default="results")
    ap.add_argument("--force", action="store_true", help="chạy lại cả job đã có kết quả")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    jobs = [(m, mode, algo, seed, var, max(100, int(DEFAULT_EPISODES[m] * args.scale)), args.out, args.force)
            for m in args.maps for mode in args.modes for algo in args.algos
            for var in args.variants for seed in range(args.seeds)]
    print(f"{len(jobs)} job, {args.workers} worker")

    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_job, j): j for j in jobs}
        for fut in as_completed(futs):
            fname, final, tt = fut.result()
            done += 1
            print(f"[{done}/{len(jobs)}] {os.path.basename(fname)} | "
                  f"return={final['mean_return']:.0f} rescued={final['mean_rescued']:.0f} "
                  f"| {tt:.0f}s", flush=True)


if __name__ == "__main__":
    main()
