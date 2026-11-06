import json
import os
from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium import spaces

UP, DOWN, LEFT, RIGHT, PICKUP, DROP = range(6)
MOVES = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}
MAP_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "maps.json")

# Reward mặc định (ghi đè bằng reward_cfg để làm thí nghiệm)
REWARD = dict(step=-1, invalid=-3, hazard=-10, pickup=5, rescue_base=10, all_bonus=50)


def load_maps(path=MAP_FILE):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class RescueEnv(gym.Env):

    def __init__(self, map_name="map1", map_file=MAP_FILE,
                 reward_mode="A", reward_cfg=None, max_steps=None):
        super().__init__()
        cfg = load_maps(map_file)[map_name]
        self.reward_mode = reward_mode
        self.R = {**REWARD, **(reward_cfg or {})}
        self.grid = cfg["grid"]
        self.H, self.W = len(self.grid), len(self.grid[0])
        self.init_med = cfg.get("medkits") or 0
        self.max_steps = max_steps or cfg["max_steps"]

        # Gom tọa độ theo ký tự
        cells = {}
        for r, row in enumerate(self.grid):
            for c, ch in enumerate(row):
                cells.setdefault(ch, []).append((r, c))
        self.walls = set(cells.get("#", []))
        self.hazards = set(cells.get("D", []))
        self.start, self.base = cells["S"][0], cells["G"][0]
        self.supply = cells["M"][0] if "M" in cells else None
        self.victim_pos = {p: i for i, p in enumerate(cells.get("V", []))}
        self.N = len(self.victim_pos)

        self.victims = cfg.get("victims") or [dict(value=1, needs_medkit=False)] * self.N
        assert len(self.victims) == self.N, "số 'victims' != số ký tự V"

        self.K = self.init_med + (1 if self.supply else 0)        # medkit tối đa
        # Mỗi thành phần của state và số giá trị của nó: (ô, mask, carry, med, supply_used)
        self.dims = [self.H * self.W, 2 ** self.N, self.N + 1, self.K + 1, 2]
        self.n_states = int(np.prod(self.dims))
        self.n_actions = 6
        self.observation_space = spaces.Discrete(self.n_states)
        self.action_space = spaces.Discrete(self.n_actions)

    def _encode(self):
        vals = [self.pos[0] * self.W + self.pos[1],
                self.mask, self.carry, self.med, self.supply_used]
        s = 0
        for v, d in zip(vals, self.dims):
            s = s * d + v
        return s

    def decode(self, s):
        vals = []
        for d in reversed(self.dims):
            s, v = divmod(s, d)
            vals.append(v)
        supply_used, med, carry, mask, cell = vals
        return dict(pos=divmod(cell, self.W), mask=mask, carry=carry,
                    med=med, supply_used=supply_used)

    def state_info(self):
        return dict(pos=self.pos, mask=self.mask, carry=self.carry,
                    med=self.med, supply_used=self.supply_used)

    def _load(self, d):
        self.pos, self.mask, self.carry = d["pos"], d["mask"], d["carry"]
        self.med, self.supply_used = d["med"], d["supply_used"]


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._load(dict(pos=self.start, mask=0, carry=0,
                        med=self.init_med, supply_used=0))
        self.t = 0
        return self._encode(), self.state_info()

    def _can_pickup(self):
        i = self.victim_pos.get(self.pos)
        return (i is not None and self.carry == 0 and not (self.mask >> i) & 1
                and (not self.victims[i]["needs_medkit"] or self.med > 0))

    def step(self, a):
        R = self.R
        reward = R["step"]
        self.t += 1

        if a in MOVES:
            nr, nc = self.pos[0] + MOVES[a][0], self.pos[1] + MOVES[a][1]
            if not (0 <= nr < self.H and 0 <= nc < self.W) or (nr, nc) in self.walls:
                reward += R["invalid"]                    # đụng tường: đứng yên
            else:
                self.pos = (nr, nc)
                if self.pos in self.hazards:
                    reward += R["hazard"]
                if self.pos == self.supply and not self.supply_used:
                    self.med += 1                         # tự động lấy medkit
                    self.supply_used = 1

        elif a == PICKUP:
            if self._can_pickup():
                i = self.victim_pos[self.pos]
                self.carry = i + 1
                self.med -= self.victims[i]["needs_medkit"]
                reward += R["pickup"]
            else:
                reward += R["invalid"]

        else:  # DROP
            if self.pos == self.base and self.carry > 0:
                i = self.carry - 1
                self.mask |= 1 << i
                self.carry = 0
                value = self.victims[i]["value"] if self.reward_mode == "B" else 1
                reward += R["rescue_base"] * value
            else:
                reward += R["invalid"]

        terminated = self.mask == 2 ** self.N - 1
        if terminated:
            reward += R["all_bonus"]
        truncated = not terminated and self.t >= self.max_steps
        return self._encode(), reward, terminated, truncated, self.state_info()

    def valid_actions(self):
        acts = [a for a, (dr, dc) in MOVES.items()
                if 0 <= self.pos[0] + dr < self.H and 0 <= self.pos[1] + dc < self.W
                and (self.pos[0] + dr, self.pos[1] + dc) not in self.walls]
        if self._can_pickup():
            acts.append(PICKUP)
        if self.pos == self.base and self.carry > 0:
            acts.append(DROP)
        return acts

    def render(self):
        rows = []
        for r in range(self.H):
            line = ""
            for c in range(self.W):
                ch = self.grid[r][c]
                if ch == "V":
                    i = self.victim_pos[(r, c)]
                    ch = "." if (self.mask >> i) & 1 else str(i)
                elif ch == "S" or (ch == "M" and self.supply_used):
                    ch = "."
                line += "@" if (r, c) == self.pos else ch
            rows.append(line)
        txt = "\n".join(rows) + (f"\nt={self.t} carry={self.carry} med={self.med} "
                                 f"mask={self.mask:0{self.N}b}")
        print(txt)
        return txt

    def optimal_steps(self):
        self.reset()
        start = self._encode()
        dist, q = {start: 0}, deque([start])
        while q:
            s = q.popleft()
            for a in range(6):
                self._load(self.decode(s))
                self.t = 0
                s2, _, term, _, _ = self.step(a)
                if term:
                    self.reset()
                    return dist[s] + 1
                if s2 not in dist:
                    dist[s2] = dist[s] + 1
                    q.append(s2)
        self.reset()
        return None


if __name__ == "__main__":
    maps = load_maps()
    for name in maps:
        e = RescueEnv(name)
        print(f"{name}: {e.H}x{e.W}, N={e.N}, |S|={e.n_states}, "
              f"max_steps={e.max_steps}, tối thiểu={e.optimal_steps()} bước")

    # Test encode/decode
    e = RescueEnv("map3")
    e.reset(seed=0)
    for _ in range(300):
        s, *_ = e.step(e.action_space.sample())
        assert e.decode(s) == e.state_info()
    print("encode/decode OK")

    # Baseline random policy
    for name in maps:
        e = RescueEnv(name)
        rets, saved = [], []
        for ep in range(200):
            e.reset(seed=ep)
            done, G = False, 0
            while not done:
                _, r, te, tr, _ = e.step(e.action_space.sample())
                G += r
                done = te or tr
            rets.append(G)
            saved.append(bin(e.mask).count("1"))
        print(f"{name}: random return={np.mean(rets):.1f}, cứu {np.mean(saved):.2f}/{e.N}")