from collections import deque
import numpy as np

from rescue_mission import RescueEnv, ACTION_NAMES


def epsilon_at(ep, episodes, eps_start, eps_end, decay_frac):
    """Epsilon giảm tuyến tính trong decay_frac * episodes episode đầu, sau đó giữ eps_end."""
    n = max(1, int(decay_frac * episodes))
    if ep >= n:
        return eps_end
    return eps_start + (eps_end - eps_start) * ep / n


def n_rescued(env):
    return bin(env.mask).count("1")


def eps_greedy(Q, s, eps, rng, n_actions):
    if rng.random() < eps:
        return int(rng.integers(n_actions))
    q = Q[s]
    # phá hòa ngẫu nhiên giữa các action có Q lớn nhất
    best = np.flatnonzero(q == q.max())
    return int(best[0]) if len(best) == 1 else int(rng.choice(best))


# ----------------------------------------------------------------------
# Q-learning
# ----------------------------------------------------------------------
class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99,
                 eps_start=1.0, eps_end=0.05, decay_frac=0.8, seed=0):
        self.env = env
        self.alpha, self.gamma = alpha, gamma
        self.eps_start, self.eps_end, self.decay_frac = eps_start, eps_end, decay_frac
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.Q = np.zeros((env.n_states, env.n_actions))

    def train(self, episodes, log_every=0):
        """Trả về dict log theo episode: return, số nạn nhân cứu, số bước."""
        env, Q = self.env, self.Q
        log = dict(returns=[], rescued=[], steps=[], eps=[])
        for ep in range(episodes):
            eps = epsilon_at(ep, episodes, self.eps_start, self.eps_end, self.decay_frac)
            s, _ = env.reset(seed=self.seed * 1_000_003 + ep)
            G, done, t = 0.0, False, 0
            while not done:
                a = eps_greedy(Q, s, eps, self.rng, env.n_actions)
                s2, r, terminated, truncated, _ = env.step(a)
                # terminated: không bootstrap; truncated (hết giờ): vẫn bootstrap
                target = r if terminated else r + self.gamma * Q[s2].max()
                Q[s, a] += self.alpha * (target - Q[s, a])
                s, G, t = s2, G + r, t + 1
                done = terminated or truncated
            log["returns"].append(G)
            log["rescued"].append(n_rescued(env))
            log["steps"].append(t)
            log["eps"].append(eps)
            if log_every and (ep + 1) % log_every == 0:
                m = slice(-log_every, None)
                print(f"[Q-learning] ep {ep+1:6d} | eps={eps:.2f} | "
                      f"return={np.mean(log['returns'][m]):8.1f} | "
                      f"rescued={np.mean(log['rescued'][m]):.2f}/{env.N}")
        return log

    def policy(self):
        return np.argmax(self.Q, axis=1)


# ----------------------------------------------------------------------
# Monte Carlo control (on-policy, first-visit, epsilon-soft)
# ----------------------------------------------------------------------
class MonteCarloAgent:
    def __init__(self, env, gamma=0.99, eps_start=1.0, eps_end=0.05,
                 decay_frac=0.8, seed=0):
        self.env = env
        self.gamma = gamma
        self.eps_start, self.eps_end, self.decay_frac = eps_start, eps_end, decay_frac
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.Q = np.zeros((env.n_states, env.n_actions))
        # số lần (s,a) được cập nhật (đặt tên counts để không nhầm với env.N = số nạn nhân)
        self.counts = np.zeros((env.n_states, env.n_actions))

    def train(self, episodes, log_every=0):
        env, Q, counts = self.env, self.Q, self.counts
        log = dict(returns=[], rescued=[], steps=[], eps=[])
        for ep in range(episodes):
            eps = epsilon_at(ep, episodes, self.eps_start, self.eps_end, self.decay_frac)
            s, _ = env.reset(seed=self.seed * 1_000_003 + ep)
            traj, G_ep, done = [], 0.0, False
            while not done:
                a = eps_greedy(Q, s, eps, self.rng, env.n_actions)
                s2, r, terminated, truncated, _ = env.step(a)
                traj.append((s, a, r))
                s, G_ep = s2, G_ep + r
                done = terminated or truncated

            # first-visit: vị trí xuất hiện đầu tiên của mỗi (s,a)
            first = {}
            for i, (st, ac, _) in enumerate(traj):
                first.setdefault((st, ac), i)
            # tính return ngược từ cuối episode
            # (nếu episode bị truncated thì G cuối coi như 0: không bootstrap)
            G = 0.0
            for i in range(len(traj) - 1, -1, -1):
                st, ac, r = traj[i]
                G = r + self.gamma * G
                if first[(st, ac)] == i:
                    counts[st, ac] += 1
                    Q[st, ac] += (G - Q[st, ac]) / counts[st, ac]   # trung bình tăng dần

            log["returns"].append(G_ep)
            log["rescued"].append(n_rescued(env))
            log["steps"].append(len(traj))
            log["eps"].append(eps)
            if log_every and (ep + 1) % log_every == 0:
                m = slice(-log_every, None)
                print(f"[Monte Carlo] ep {ep+1:6d} | eps={eps:.2f} | "
                      f"return={np.mean(log['returns'][m]):8.1f} | "
                      f"rescued={np.mean(log['rescued'][m]):.2f}/{env.N}")
        return log

    def policy(self):
        return np.argmax(self.Q, axis=1)


# ----------------------------------------------------------------------
# Policy Iteration (có mô hình, chỉ nên dùng trên map nhỏ)
# ----------------------------------------------------------------------
def _set_state(env, s):
    d = env.decode(s)
    env.pos, env.mask, env.carry = d["pos"], d["mask"], d["carry"]
    env.med, env.supply_used, env.t = d["med"], d["supply_used"], 0


def build_model(env):
    """Duyệt các state tới được từ state đầu; trả về trans[s][a] = (s', r, terminated).
    State kết thúc (đã cứu đủ) không được mở rộng thêm."""
    saved_max = env.max_steps
    full = 2 ** env.N - 1
    env.max_steps = 10 ** 9              # tắt truncation khi dựng mô hình
    try:
        s0, _ = env.reset()
        trans, seen, q = {}, {s0}, deque([s0])
        while q:
            s = q.popleft()
            if env.decode(s)["mask"] == full:        # terminal: không mở rộng
                trans[s] = [(s, 0.0, True)] * env.n_actions
                continue
            trans[s] = []
            for a in range(env.n_actions):
                _set_state(env, s)
                s2, r, te, _, _ = env.step(a)
                trans[s].append((s2, r, te))
                if s2 not in seen:
                    seen.add(s2)
                    q.append(s2)
    finally:
        env.max_steps = saved_max
        env.reset()
    return s0, trans


def policy_iteration(env, gamma=0.99, theta=1e-8, max_iter=200):
    """Trả về (Q đầy đủ kích thước n_states x n_actions, số vòng lặp, số state tới được).
    Cài đặt vector hóa bằng numpy nên nhanh hơn nhiều so với vòng for thuần Python."""
    s0, trans = build_model(env)
    states = list(trans.keys())
    idx = {s: i for i, s in enumerate(states)}
    n, A = len(states), env.n_actions

    nxt = np.zeros((n, A), dtype=np.int64)
    rew = np.zeros((n, A))
    term = np.zeros((n, A), dtype=bool)
    for s, row in trans.items():
        i = idx[s]
        for a, (s2, r, te) in enumerate(row):
            nxt[i, a], rew[i, a], term[i, a] = idx[s2], r, te

    ar = np.arange(n)
    pi = np.zeros(n, dtype=np.int64)
    V = np.zeros(n)

    def q_matrix():
        return rew + gamma * np.where(term, 0.0, V[nxt])

    it = 0
    for it in range(1, max_iter + 1):
        # 1) đánh giá policy (lặp đồng bộ cho tới khi hội tụ)
        while True:
            Vn = q_matrix()[ar, pi]
            delta = np.abs(Vn - V).max()
            V = Vn
            if delta < theta:
                break
        # 2) cải thiện policy
        Qm = q_matrix()
        best = np.argmax(Qm, axis=1)
        better = Qm[ar, best] > Qm[ar, pi] + 1e-9
        if not better.any():
            break
        pi[better] = best[better]

    Qm = q_matrix()
    Q = np.zeros((env.n_states, env.n_actions))
    for s, i in idx.items():
        Q[s] = Qm[i]
    return Q, it, n


# ----------------------------------------------------------------------
# Đánh giá và trích xuất đường đi
# ----------------------------------------------------------------------
def evaluate(env, Q, episodes=1, seed=12345):
    """Chạy policy greedy theo Q. Trả về thống kê.
    Env và policy greedy đều deterministic nên 1 episode là đủ; tăng `episodes`
    chỉ có ý nghĩa nếu sau này env có yếu tố ngẫu nhiên.
    Lưu ý: state chưa từng được thăm có Q = 0 nên argmax sẽ chọn action 0 (Lên)."""
    rets, rescued, steps, success = [], [], [], []
    for ep in range(episodes):
        s, _ = env.reset(seed=seed + ep)
        done, G, t, te = False, 0.0, 0, False
        while not done:
            s, r, te, tr, _ = env.step(int(np.argmax(Q[s])))
            G, t = G + r, t + 1
            done = te or tr
        rets.append(G); rescued.append(n_rescued(env))
        steps.append(t); success.append(float(te))
    return dict(mean_return=float(np.mean(rets)), mean_rescued=float(np.mean(rescued)),
                success_rate=float(np.mean(success)), mean_steps=float(np.mean(steps)))


def rollout(env, Q, max_len=None, verbose=False):
    """Một episode greedy. Trả về đường đi, thứ tự cứu và tổng reward."""
    s, _ = env.reset(seed=0)
    path, order, actions, G, done = [env.pos], [], [], 0.0, False
    te = False                                   # tránh NameError nếu vòng lặp không chạy
    limit = max_len or env.max_steps
    while not done and len(actions) < limit:
        a = int(np.argmax(Q[s]))
        before = env.mask
        s, r, te, tr, _ = env.step(a)
        actions.append(ACTION_NAMES[a]); path.append(env.pos); G += r
        new = env.mask ^ before
        if new:
            order.append(new.bit_length() - 1)
        done = te or tr
        if verbose:
            env.render(); print("-> action:", ACTION_NAMES[a], "reward:", r, "\n")
    return dict(path=path, actions=actions, order=order, total_return=G,
                steps=len(actions), terminated=bool(te))


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "map1"
    mode = sys.argv[2] if len(sys.argv) > 2 else "A"
    env = RescueEnv(name, reward_mode=mode)
    print(f"{name} mode {mode}: |S|={env.n_states}, |A|={env.n_actions}")

    ql = QLearningAgent(env, seed=0)
    ql.train(10000, log_every=2000)
    print("Q-learning :", evaluate(env, ql.Q), rollout(env, ql.Q)["order"])

    mc = MonteCarloAgent(env, seed=0)
    mc.train(20000, log_every=4000)
    print("Monte Carlo:", evaluate(env, mc.Q), rollout(env, mc.Q)["order"])

    if env.n_states <= 5000:
        Qpi, it, n = policy_iteration(env)
        print(f"Policy Iter: ({it} vòng, {n} state tới được)", evaluate(env, Qpi),
              rollout(env, Qpi)["order"])
