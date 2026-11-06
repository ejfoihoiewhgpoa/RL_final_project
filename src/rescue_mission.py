import json
import os
import warnings
from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium import spaces

UP, DOWN, LEFT, RIGHT, PICKUP, DROP = range(6)
ACTION_NAMES = ["Lên", "Xuống", "Trái", "Phải", "Pickup", "Drop"]
MOVES = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}

# maps.json nằm cùng thư mục với file này (không hardcode đường dẫn tuyệt đối)
DEFAULT_MAP_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "maps.json")

# Ký hiệu hiển thị nạn nhân: 0-9 rồi a-z (tránh lệch grid khi có >= 10 nạn nhân)
VICTIM_SYMBOLS = "0123456789abcdefghijklmnopqrstuvwxyz"

# Tham số reward (đổi ở đây để làm thí nghiệm reward)
DEFAULT_REWARD = dict(
    step=-1.0,         # mỗi bước
    invalid=-5.0,      # đụng tường / Pickup, Drop không hợp lệ
    hazard=-10.0,      # bước vào ô D
    pickup=0.0,        # Pickup thành công
    rescue_base=50.0,  # reward khi Drop nạn nhân về G (mode B nhân thêm value)
    all_bonus=100.0,   # bonus khi cứu đủ
)


def load_maps(path=DEFAULT_MAP_FILE):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class RescueEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, map_name="map1", map_file=DEFAULT_MAP_FILE,
                 reward_mode="A", reward_cfg=None, max_steps=None, render_mode=None):
        """
        map_name   : key trong file JSON (hoặc truyền thẳng 1 dict cấu hình)
        reward_mode: 'A' = cứu gần trước (R_i như nhau)
                     'B' = ưu tiên giá trị cao (R_i = rescue_base * value_i)
        """
        super().__init__()
        assert render_mode in (None, "ansi"), "render_mode chỉ hỗ trợ None hoặc 'ansi'"
        cfg = load_maps(map_file)[map_name] if isinstance(map_name, str) else map_name
        self.map_name = map_name if isinstance(map_name, str) else "custom"
        self.reward_mode = reward_mode
        self.R = {**DEFAULT_REWARD, **(reward_cfg or {})}
        self.render_mode = render_mode

        # ---- parse grid ----
        self.layout = cfg["grid"]
        self.H, self.W = len(self.layout), len(self.layout[0])
        self.init_med = cfg.get("medkits") or 0
        self.max_steps = max_steps or cfg["max_steps"]

        self.walls, self.hazards = set(), set()
        self.victim_pos = {}                 # (r,c) -> index nạn nhân
        self.supply = None
        self.start = self.base = None
        for r, row in enumerate(self.layout):
            assert len(row) == self.W, "các hàng của grid phải cùng độ dài"
            for c, ch in enumerate(row):
                if ch == "#":
                    self.walls.add((r, c))
                elif ch == "D":
                    self.hazards.add((r, c))
                elif ch == "S":
                    assert self.start is None, "mỗi map chỉ được có đúng 1 ô S"
                    self.start = (r, c)
                elif ch == "G":
                    assert self.base is None, "mỗi map chỉ được có đúng 1 ô G"
                    self.base = (r, c)
                elif ch == "M":
                    assert self.supply is None, "mỗi map chỉ được có tối đa 1 ô M"
                    self.supply = (r, c)
                elif ch == "V":
                    self.victim_pos[(r, c)] = len(self.victim_pos)
        assert self.start and self.base, "map cần có S và G"
        self.N = len(self.victim_pos)
        assert self.N >= 1, "map cần có ít nhất 1 nạn nhân V"

        # ---- thuộc tính nạn nhân (tùy chọn trong JSON) ----
        self.victims = cfg.get("victims") or [
            dict(value=1, needs_medkit=False) for _ in range(self.N)]
        assert len(self.victims) == self.N, "số phần tử 'victims' != số ký tự V trong grid"

        # ---- không gian state / action ----
        self.K = self.init_med + (1 if self.supply else 0)   # medkit tối đa
        need = sum(1 for v in self.victims if v["needs_medkit"])
        if need > self.K:
            warnings.warn(f"{self.map_name}: cần {need} medkit nhưng tối đa chỉ có "
                          f"{self.K} -> không thể cứu hết nạn nhân.")
        # GIỮ NGUYÊN công thức cũ (supply_used luôn có 2 giá trị) để Q-table cũ vẫn khớp
        self.n_states = (self.H * self.W * (2 ** self.N)
                         * (self.N + 1) * (self.K + 1) * 2)
        self.observation_space = spaces.Discrete(self.n_states)
        self.action_space = spaces.Discrete(6)
        self.n_actions = 6

        # khởi tạo trạng thái để các hàm gọi trước reset() không bị lỗi thiếu thuộc tính
        self._load(dict(pos=self.start, mask=0, carry=0,
                        med=self.init_med, supply_used=0))
        self.t = 0

    # ------------------------------------------------------------------
    def _encode(self):
        r, c = self.pos
        s = r * self.W + c
        s = s * (2 ** self.N) + self.mask
        s = s * (self.N + 1) + self.carry
        s = s * (self.K + 1) + self.med
        s = s * 2 + self.supply_used
        return s

    def decode(self, s):
        s, sup = divmod(s, 2)
        s, med = divmod(s, self.K + 1)
        s, carry = divmod(s, self.N + 1)
        s, mask = divmod(s, 2 ** self.N)
        r, c = divmod(s, self.W)
        return dict(pos=(r, c), mask=mask, carry=carry, med=med, supply_used=sup)

    def state_info(self):
        return dict(pos=self.pos, mask=self.mask, carry=self.carry,
                    med=self.med, supply_used=self.supply_used)

    def _load(self, d):
        self.pos, self.mask, self.carry = tuple(d["pos"]), d["mask"], d["carry"]
        self.med, self.supply_used = d["med"], d["supply_used"]

    def _rescue_reward(self, i):
        if self.reward_mode == "B":
            return self.R["rescue_base"] * self.victims[i]["value"]
        return self.R["rescue_base"]

    def _can_pickup(self):
        i = self.victim_pos.get(self.pos)
        return (i is not None and self.carry == 0
                and not (self.mask >> i) & 1
                and (not self.victims[i]["needs_medkit"] or self.med > 0))

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._load(dict(pos=self.start, mask=0, carry=0,
                        med=self.init_med, supply_used=0))
        self.t = 0
        return self._encode(), self.state_info()

    def step(self, a):
        reward = self.R["step"]
        self.t += 1
        r, c = self.pos

        if a in MOVES:
            dr, dc = MOVES[a]
            nr, nc = r + dr, c + dc
            if (not (0 <= nr < self.H and 0 <= nc < self.W)
                    or (nr, nc) in self.walls):
                reward += self.R["invalid"]               # đụng tường: đứng yên
            else:
                self.pos = (nr, nc)
                if self.pos in self.hazards:
                    reward += self.R["hazard"]
                if self.pos == self.supply and not self.supply_used:
                    self.med += 1                         # lấy supply tự động
                    self.supply_used = 1

        elif a == PICKUP:
            if self._can_pickup():
                i = self.victim_pos[self.pos]
                self.carry = i + 1
                if self.victims[i]["needs_medkit"]:
                    self.med -= 1
                reward += self.R["pickup"]
            else:
                reward += self.R["invalid"]

        elif a == DROP:
            if self.pos == self.base and self.carry > 0:
                i = self.carry - 1
                self.mask |= 1 << i
                self.carry = 0
                reward += self._rescue_reward(i)
            else:
                reward += self.R["invalid"]

        terminated = self.mask == (2 ** self.N) - 1
        if terminated:
            reward += self.R["all_bonus"]
        truncated = (not terminated) and self.t >= self.max_steps
        return self._encode(), reward, terminated, truncated, self.state_info()

    def valid_actions(self):
        acts = []
        r, c = self.pos
        for a, (dr, dc) in MOVES.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.H and 0 <= nc < self.W and (nr, nc) not in self.walls:
                acts.append(a)
        if self._can_pickup():
            acts.append(PICKUP)
        if self.pos == self.base and self.carry > 0:
            acts.append(DROP)
        return acts

    def render(self):
        carried = self.carry - 1 if self.carry > 0 else None
        rows = []
        for r in range(self.H):
            line = ""
            for c in range(self.W):
                ch = self.layout[r][c]
                if ch == "V":
                    i = self.victim_pos[(r, c)]
                    if (self.mask >> i) & 1 or i == carried:
                        ch = "."                       # đã cứu hoặc đang được bế
                    else:
                        ch = VICTIM_SYMBOLS[i] if i < len(VICTIM_SYMBOLS) else "?"
                elif ch == "M" and self.supply_used:
                    ch = "."
                elif ch == "S":
                    ch = "."
                if (r, c) == self.pos:
                    ch = "@"
                line += ch
            rows.append(line)
        txt = "\n".join(rows)
        txt += (f"\nt={self.t} carry={self.carry} med={self.med} "
                f"mask={self.mask:0{self.N}b}")
        print(txt)
        return txt

    # ------------------------------------------------------------------
    def optimal_steps(self):
        """BFS trên không gian state: số bước ít nhất để cứu đủ (None nếu không thể).
        Dùng để chọn max_steps và kiểm tra map có lời giải.
        Không làm thay đổi trạng thái hiện tại của env."""
        saved = (self.state_info(), self.t)
        try:
            self._load(dict(pos=self.start, mask=0, carry=0,
                            med=self.init_med, supply_used=0))
            start = self._encode()
            dist = {start: 0}
            q = deque([start])
            while q:
                s = q.popleft()
                for a in range(6):
                    self._load(self.decode(s))
                    self.t = 0
                    s2, _, term, _, _ = self.step(a)
                    if s2 == s:
                        continue
                    if term:
                        return dist[s] + 1
                    if s2 not in dist:
                        dist[s2] = dist[s] + 1
                        q.append(s2)
            return None
        finally:
            self._load(saved[0])
            self.t = saved[1]


if __name__ == "__main__":
    maps = load_maps()
    print("== State / action space, lời giải tối ưu ==")
    for name in maps:
        e = RescueEnv(name)
        k = e.optimal_steps()
        print(f"{name}: grid {e.H}x{e.W}, N={e.N}, K_max={e.K}, |S|={e.n_states}, "
              f"|A|={e.n_actions}, max_steps={e.max_steps}, "
              f"tối thiểu={k if k is not None else 'KHÔNG GIẢI ĐƯỢC'} bước")
        if k is not None and k > e.max_steps:
            print(f"  [!] {name}: max_steps ({e.max_steps}) < số bước tối thiểu ({k})")

    # Test encode/decode (có seed để tái lập)
    e = RescueEnv("map3")
    e.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(300):
        s, *_ = e.step(int(rng.integers(e.n_actions)))
        assert e.decode(s) == e.state_info()
    print("encode/decode OK")

    print("\n== Random policy (200 episode) ==")
    for name in maps:
        e = RescueEnv(name)
        rets, saved = [], []
        for ep in range(200):
            e.reset(seed=ep)
            rng = np.random.default_rng(ep)
            done, G = False, 0
            while not done:
                _, r, te, tr, _ = e.step(int(rng.integers(e.n_actions)))
                G += r
                done = te or tr
            rets.append(G)
            saved.append(bin(e.mask).count("1"))
        print(f"{name}: mean return={np.mean(rets):8.1f} | "
              f"mean rescued={np.mean(saved):.2f}/{e.N}")