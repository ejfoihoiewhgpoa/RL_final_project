import argparse
import os
import numpy as np
import pygame

from rescue_mission import RescueEnv, MOVES
from agents import QLearningAgent, evaluate

EPISODES = {"map1": 20000, "map2": 50000, "map3": 60000}
QDIR = "qtables"

C_BG, C_GRID = (245, 245, 245), (190, 190, 190)
C_WALL, C_HAZ, C_BASE = (70, 70, 70), (244, 166, 166), (168, 224, 168)
C_SUP, C_START = (169, 201, 245), (255, 243, 168)
C_VIC, C_VIC_DONE = (255, 179, 71), (150, 200, 150)
C_AGENT, C_TEXT = (40, 90, 200), (30, 30, 30)


def q_path(map_name, mode):
    return os.path.join(QDIR, f"{map_name}_{mode}.npy")


def get_q(map_name, mode, retrain=False, on_msg=None):
    p = q_path(map_name, mode)
    if os.path.exists(p) and not retrain:
        return np.load(p)
    if on_msg:
        on_msg(f"Đang huấn luyện Q-learning cho {map_name} / mode {mode} ... (chỉ lần đầu)")
    env = RescueEnv(map_name, reward_mode=mode)
    agent = QLearningAgent(env, seed=0)
    agent.train(EPISODES[map_name])
    os.makedirs(QDIR, exist_ok=True)
    np.save(p, agent.Q)
    return agent.Q


def make_font(size, bold=False):
    return pygame.font.SysFont("segoeui,arial,dejavusans,freesans", size, bold=bold)


class Demo:
    def __init__(self, map_name, mode, fps, retrain=False):
        pygame.init()
        self.fps = fps
        self.map_name, self.mode = map_name, mode
        self.f_s, self.f_m, self.f_b = make_font(15), make_font(18), make_font(22, True)
        self.screen = pygame.display.set_mode((1000, 620))
        pygame.display.set_caption("Rescue Mission – demo policy")
        self.load(map_name, mode, retrain)

    # ------------------------------------------------------------------
    def message(self, text):
        self.screen.fill(C_BG)
        self.screen.blit(self.f_b.render(text, True, C_TEXT), (30, 280))
        pygame.display.flip()
        pygame.event.pump()

    def load(self, map_name, mode, retrain=False):
        self.map_name, self.mode = map_name, mode
        self.env = RescueEnv(map_name, reward_mode=mode)
        self.Q = get_q(map_name, mode, retrain, self.message)
        e = self.env
        self.cell = min(88, 560 // max(e.H, e.W))
        self.ox, self.oy = 20, 20
        self.restart()

    def restart(self):
        self.s, _ = self.env.reset(seed=0)
        self.total, self.last_action, self.done, self.status = 0.0, "-", False, ""
        self.paused, self.timer = False, 0

    def step(self):
        if self.done:
            return
        a = int(np.argmax(self.Q[self.s]))
        self.s, r, te, tr, _ = self.env.step(a)
        self.total += r
        self.last_action = MOVES[a]
        if te:
            self.done, self.status = True, "HOÀN THÀNH: đã cứu hết nạn nhân"
        elif tr:
            self.done, self.status = True, "HẾT GIỜ (max_steps)"

    # ------------------------------------------------------------------
    def draw_board(self):
        e, cs = self.env, self.cell
        for r in range(e.H):
            for c in range(e.W):
                ch = e.layout[r][c]
                color = {"#": C_WALL, "D": C_HAZ, "G": C_BASE, "M": C_SUP,
                         "S": C_START}.get(ch, (255, 255, 255))
                if ch == "M" and e.supply_used:
                    color = (225, 232, 245)
                rect = pygame.Rect(self.ox + c * cs, self.oy + r * cs, cs, cs)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, C_GRID, rect, 1)
                label = {"D": "!", "G": "G", "S": "S", "M": "+" if not e.supply_used else ""}.get(ch)
                if label:
                    self.screen.blit(self.f_s.render(label, True, C_TEXT), (rect.x + 5, rect.y + 3))
        for (r, c), i in e.victim_pos.items():
            done = (e.mask >> i) & 1
            cx, cy = self.ox + c * cs + cs // 2, self.oy + r * cs + cs // 2
            if done:                                       # đã được đưa về G: mờ
                pygame.draw.circle(self.screen, C_VIC_DONE, (cx, cy), cs // 3, 2)
                continue
            if e.carry == i + 1:                           # đang mang: vẽ trên agent
                continue
            pygame.draw.circle(self.screen, C_VIC, (cx, cy), cs // 3)
            pygame.draw.circle(self.screen, (0, 0, 0), (cx, cy), cs // 3, 2)
            tag = f"V{i}" + ("+" if e.victims[i]["needs_medkit"] else "")
            t1 = self.f_s.render(tag, True, C_TEXT)
            t2 = self.f_s.render(f"v={e.victims[i]['value']}", True, C_TEXT)
            self.screen.blit(t1, t1.get_rect(center=(cx, cy - 7)))
            self.screen.blit(t2, t2.get_rect(center=(cx, cy + 9)))
        r, c = e.pos
        ax, ay = self.ox + c * cs + cs // 2, self.oy + r * cs + cs // 2
        pygame.draw.circle(self.screen, C_AGENT, (ax, ay), cs // 4)
        pygame.draw.circle(self.screen, (255, 255, 255), (ax, ay), cs // 4, 2)
        if e.carry:
            pygame.draw.circle(self.screen, C_VIC, (ax, ay - cs // 4), cs // 7)
            pygame.draw.circle(self.screen, (0, 0, 0), (ax, ay - cs // 4), cs // 7, 1)
            self.screen.blit(self.f_s.render(f"V{e.carry-1}", True, C_TEXT), (ax + cs // 5, ay - cs // 3))

    def draw_panel(self):
        e = self.env
        x = self.ox + e.W * self.cell + 30
        y = 20
        def line(text, font=None, color=C_TEXT, dy=26):
            nonlocal y
            self.screen.blit((font or self.f_m).render(text, True, color), (x, y))
            y += dy
        line(f"Rescue Mission – {self.map_name}", self.f_b, dy=34)
        line(f"Mode {self.mode}: " + ("cứu gần trước" if self.mode == "A" else "ưu tiên giá trị cao"))
        line(f"Bước: {e.t} / {e.max_steps}")
        line(f"Medkit còn lại: {e.med}   (ô M đã dùng: {'có' if e.supply_used else 'chưa'})")
        carry = f"V{e.carry-1}" if e.carry else "không"
        line(f"Đang mang: {carry}")
        saved = [f"V{i}" for i in range(e.N) if (e.mask >> i) & 1]
        line(f"Đã cứu: {', '.join(saved) if saved else '-'}  ({len(saved)}/{e.N})")
        line(f"Action vừa rồi: {self.last_action}")
        line(f"Tổng reward: {self.total:.0f}", dy=34)
        if self.status:
            ok = self.status.startswith("HOÀN")
            line(self.status, self.f_m, (20, 130, 20) if ok else (180, 30, 30), dy=34)
        for t in ("Phím:", "SPACE  tạm dừng / chạy", "→  đi 1 bước      R  chạy lại",
                  "A / B  đổi mode      1 / 2 / 3  đổi map", "↑ / ↓  tốc độ (" + f"{self.fps} bước/giây)",
                  "ESC  thoát"):
            line(t, self.f_s, (90, 90, 90), dy=22)
        y += 6
        line("Chú giải: ! nguy hiểm | + ô medkit | G điểm an toàn", self.f_s, (90, 90, 90), dy=20)
        line("V+ = nạn nhân cần medkit | v = giá trị", self.f_s, (90, 90, 90), dy=20)

    # ------------------------------------------------------------------
    def run(self):
        clock = pygame.time.Clock()
        running = True
        while running:
            dt = clock.tick(60)
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.KEYDOWN:
                    k = ev.key
                    if k == pygame.K_ESCAPE:
                        running = False
                    elif k == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif k == pygame.K_RIGHT:
                        self.paused = True; self.step()
                    elif k == pygame.K_r:
                        self.restart()
                    elif k == pygame.K_a:
                        self.load(self.map_name, "A")
                    elif k == pygame.K_b:
                        self.load(self.map_name, "B")
                    elif k in (pygame.K_1, pygame.K_2, pygame.K_3):
                        self.load({pygame.K_1: "map1", pygame.K_2: "map2", pygame.K_3: "map3"}[k], self.mode)
                    elif k == pygame.K_UP:
                        self.fps = min(60, self.fps + 1)
                    elif k == pygame.K_DOWN:
                        self.fps = max(1, self.fps - 1)
            if not self.paused and not self.done:
                self.timer += dt
                if self.timer >= 1000 / self.fps:
                    self.timer = 0
                    self.step()
            self.screen.fill(C_BG)
            self.draw_board()
            self.draw_panel()
            pygame.display.flip()
        pygame.quit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="map1", choices=list(EPISODES))
    ap.add_argument("--mode", default="A", choices=["A", "B"])
    ap.add_argument("--fps", type=int, default=4, help="số bước agent đi mỗi giây")
    ap.add_argument("--retrain", action="store_true", help="huấn luyện lại, ghi đè Q-table")
    ap.add_argument("--prepare", action="store_true", help="huấn luyện sẵn mọi map/mode rồi thoát")
    ap.add_argument("--screenshot", help="(kiểm thử) chạy N bước, lưu ảnh rồi thoát")
    ap.add_argument("--steps", type=int, default=30)
    args = ap.parse_args()

    if args.prepare:
        for m in EPISODES:
            for mode in "AB":
                print(f"Huấn luyện {m} / {mode} ...", flush=True)
                Q = get_q(m, mode, args.retrain)
                env = RescueEnv(m, reward_mode=mode)
                print("  ->", evaluate(env, Q, episodes=1))
        return

    demo = Demo(args.map, args.mode, args.fps, args.retrain)
    if args.screenshot:
        for _ in range(args.steps):
            demo.step()
        demo.screen.fill(C_BG); demo.draw_board(); demo.draw_panel()
        pygame.image.save(demo.screen, args.screenshot)
        return
    demo.run()


if __name__ == "__main__":
    main()
