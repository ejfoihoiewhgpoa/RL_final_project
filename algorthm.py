from collections import deque
import numpy as np

from MDP import RescueEnv, MOVES

def eps_greedy(Q, s, eps, rng, n_actions):
    if rng.random() < eps:
        return int(rng.integers(n_actions))
    best = np.flatnonzero(Q[s] == Q[s].max())
    return int(rng.choice(best))

# Q-learning
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

# Monte Carlo control (on-policy, first-visit, epsilon-soft)
class MonteCarloAgent:
    def __init__(self, env, gamma=0.99, eps_start=1.0, eps_end=0.05,
                 decay_frac=0.8, seed=0):
        self.env = env
        self.gamma = gamma
        self.eps_start, self.eps_end, self.decay_frac = eps_start, eps_end, decay_frac
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.Q = np.zeros((env.n_states, env.n_actions))
        self.N = np.zeros((env.n_states, env.n_actions))   # số lần (s,a) được cập nhật

    def train(self, episodes, log_every=0):
        env, Q, N = self.env, self.Q, self.N
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
            G = 0.0
            for i in range(len(traj) - 1, -1, -1):
                st, ac, r = traj[i]
                G = r + self.gamma * G
                if first[(st, ac)] == i:
                    N[st, ac] += 1
                    Q[st, ac] += (G - Q[st, ac]) / N[st, ac]   # trung bình tăng dần

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
    """Duyệt các state tới được từ state đầu; trả về trans[s][a] = (s', r, terminated)."""
    saved_max = env.max_steps
    env.max_steps = 10 ** 9              # tắt truncation khi dựng mô hình
    s0, _ = env.reset()
    trans, seen, q = {}, {s0}, deque([s0])
    while q:
        s = q.popleft()
        trans[s] = []
        for a in range(env.n_actions):
            _set_state(env, s)
            s2, r, te, _, _ = env.step(a)
            trans[s].append((s2, r, te))
            if s2 not in seen:
                seen.add(s2)
                q.append(s2)
    env.max_steps = saved_max
    return s0, trans


def policy_iteration(env, gamma=0.99, theta=1e-8, max_iter=200):
    """Trả về (Q đầy đủ kích thước n_states x n_actions, số vòng lặp, số state tới được)."""
    s0, trans = build_model(env)
    states = list(trans.keys())
    pi = {s: 0 for s in states}
    V = {s: 0.0 for s in states}

    def q_val(s, a):
        s2, r, te = trans[s][a]
        return r + (0.0 if te else gamma * V[s2])

    it = 0
    for it in range(1, max_iter + 1):
        # 1) đánh giá policy
        while True:
            delta = 0.0
            for s in states:
                v = q_val(s, pi[s])
                delta = max(delta, abs(v - V[s]))
                V[s] = v
            if delta < theta:
                break
        # 2) cải thiện policy
        stable = True
        for s in states:
            qs = [q_val(s, a) for a in range(env.n_actions)]
            best = int(np.argmax(qs))
            if qs[best] > qs[pi[s]] + 1e-9:
                pi[s] = best
                stable = False
        if stable:
            break

    Q = np.zeros((env.n_states, env.n_actions))
    for s in states:
        for a in range(env.n_actions):
            Q[s, a] = q_val(s, a)
    return Q, it, len(states)


# ----------------------------------------------------------------------
# Đánh giá và trích xuất đường đi
# ----------------------------------------------------------------------
def evaluate(env, Q, episodes=100, seed=12345):
    """Chạy policy greedy theo Q. Trả về thống kê."""
    rets, rescued, steps, success = [], [], [], []
    for ep in range(episodes):
        s, _ = env.reset(seed=seed + ep)
        done, G, t = False, 0.0, 0
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
    limit = max_len or env.max_steps
    while not done and len(actions) < limit:
        a = int(np.argmax(Q[s]))
        before = env.mask
        s, r, te, tr, _ = env.step(a)
        actions.append(MOVES[a]); path.append(env.pos); G += r
        new = env.mask ^ before
        if new:
            order.append(new.bit_length() - 1)
        done = te or tr
        if verbose:
            env.render(); print("-> action:", MOVES[a], "reward:", r, "\n")
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