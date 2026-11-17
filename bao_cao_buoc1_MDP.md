# Đề tài 30 – Rescue Mission nhiều nạn nhân

Agent di chuyển trên lưới, cứu nhiều nạn nhân và đưa về điểm an toàn. Project cài đặt môi trường `gymnasium` tự viết, hai thuật toán **Monte Carlo control** và **Q-learning** (dạng bảng, không dùng DQN), cùng policy tối ưu tham chiếu để so sánh.

## 1. Cấu trúc thư mục

```
final_project/1/
├── src/
│   ├── rescue_mission.py     # môi trường RescueEnv (đọc map từ maps.json)
│   ├── maps.json             # map1, map2, map3
│   ├── agents.py             # QLearningAgent, MonteCarloAgent, Policy Iteration, evaluate, rollout
│   ├── train.py              # huấn luyện nhiều seed / thuật toán / reward mode
│   ├── plot_results.py       # vẽ đồ thị từ các file .pkl trong results/
│   ├── compare_policies.py   # so sánh các policy
│   ├── optimal_baseline.py   # policy tối ưu tham chiếu (Value Iteration)
│   └── demo.py               # xem agent chạy hoặc tự chơi (giao diện pygame)
├── qtables/                  # Q-table đã huấn luyện cho demo (map1_A.npy, ...)
├── results/                  # kết quả huấn luyện (.pkl) và hình
├── bao_cao_buoc1_MDP...      # báo cáo bước 1: mô hình MDP và thiết kế map
├── requirements.txt
└── README.md
```

## 2. Cài đặt

Yêu cầu Python 3.11.

```
pip install -r requirements.txt
```

Các thư viện chính: `numpy`, `gymnasium`, `matplotlib`, `pygame`.

## 3. Cách chạy

Chạy từ thư mục gốc của project.

**Huấn luyện**

```
python src/train.py                                   # chạy đầy đủ (3 map)
python src/train.py --maps map1 --seeds 3 --scale 0.2 # chạy thử nhanh
python src/train.py --variants base low_hazard        # thêm biến thể reward
```

Mỗi job lưu 1 file `.pkl` trong `results/` để `plot_results.py` đọc.

**Vẽ đồ thị**

```
python src/plot_results.py
```

**Demo và chơi thử**

```
python src/demo.py                      # agent tự chạy (map1, mode A)
python src/demo.py --map map2 --mode B
python src/demo.py --play               # tự mình chơi
python src/demo.py --prepare            # huấn luyện sẵn Q-table cho mọi map/mode
```

Lần đầu chạy agent trên một map/mode, demo sẽ huấn luyện Q-learning rồi lưu vào `qtables/` (các lần sau dùng lại). Thêm `--retrain` để huấn luyện lại.

| Phím | Tác dụng |
|---|---|
| P | đổi giữa chế độ AGENT và NGƯỜI CHƠI |
| R | chạy lại / chơi lại |
| A / B | đổi reward mode |
| 1 / 2 / 3 | đổi map |
| Mũi tên | di chuyển (chế độ người chơi) |
| SPACE, → , ↑ / ↓ | tạm dừng, đi 1 bước, đổi tốc độ (chế độ agent) |
| ESC | thoát |

## 4. Mô hình MDP (tóm tắt)

Chi tiết và lập luận xem báo cáo bước 1.

- **Môi trường**: lưới H×W gồm `S` (xuất phát), `V` (nạn nhân), `G` (điểm an toàn), `D` (nguy hiểm), `M` (ô bổ sung đúng 1 medkit, dùng một lần), `#` (tường). Mỗi lần chỉ mang được 1 người; một số nạn nhân cần medkit để cứu.
- **State**: `s = (r, c, mask, carry, med, supply_used)`
  - `mask`: bit i = 1 nếu nạn nhân i đã được đưa về G
  - `carry`: 0 là không mang ai, i+1 là đang mang nạn nhân i
  - `med`: số medkit còn lại; `supply_used`: ô M đã dùng chưa
- **Action** (6): Lên, Xuống, Trái, Phải, Pickup, Drop. Hành động không hợp lệ thì đứng yên và bị phạt.
- **Transition**: xác định (deterministic).
- **Reward**:

| Sự kiện | Reward |
|---|---|
| Mỗi bước | −1 |
| Hành động không hợp lệ | −5 |
| Bước vào ô D | −10 |
| Drop nạn nhân i về G | `+R_i` |
| Cứu đủ N nạn nhân | +100 |

- **Hai reward mode**: Mode A (cứu gần trước) có `R_i = 50`; Mode B (ưu tiên giá trị cao) có `R_i = 50 · value_i`.
- **Kết thúc**: *terminated* khi cứu hết nạn nhân; *truncated* khi hết `max_steps` (ô D chỉ bị phạt, không kết thúc episode). Discount γ = 0,99.

Giới hạn thời gian được xử lý như truncation, không đưa vào state, nên MDP gốc vẫn giữ tính Markov.

## 5. Ba map

| Map | Lưới | N | Medkit đầu | Nạn nhân cần medkit | Ô M | \|S\| | max_steps |
|---|---|---|---|---|---|---|---|
| map1 | 5×5 | 2 | – | 0 | không | 600 | 30 |
| map2 | 6×6 | 3 | 1 | 2 | có | 6.912 | 45 |
| map3 | 7×7 | 4 | 2 | 3 | có | 31.360 | 140 |

Ánh xạ với các setting của đề:

| Setting của đề | Map |
|---|---|
| 2 nạn nhân, không giới hạn nguồn lực | map1 |
| 3–4 nạn nhân, mang 1 người/lần | map2, map3 |
| Một số nạn nhân cần medkit, medkit hữu hạn | map2, map3 |
| Ô supply bổ sung 1 medkit, phải đi đường vòng | map2 (31 → 33 bước); map3 có ô supply nhưng nằm trên đường tối ưu |

Ở map3, ô `D` có đường né (mở tường ô (5,4)), nên đi qua `D` là lựa chọn của policy chứ không bắt buộc.

## 6. Kết quả tham chiếu

**Policy tối ưu (Value Iteration, γ = 0,99)**

| Map | Mode | Thứ tự cứu | Số bước | Return |
|---|---|---|---|---|
| map1 | A | V0 → V1 | 24 | 176 |
| map1 | B | V0 → V1 | 24 | 276 |
| map2 | A | V2 → V1 → V0 | 33 | 217 |
| map2 | B | V0 → V1 → V2 | 33 | 367 |
| map3 | A | V1 → V2 → V3 → V0 | 109 | 191 |
| map3 | B | V1 → V0 → V2 → V3 | 109 | 391 |

Ở map2 và map3, hai mode cho thứ tự cứu khác nhau: mode A ưu tiên nạn nhân gần, mode B ưu tiên nạn nhân giá trị cao.

**Baseline ngẫu nhiên (200 episode)**

| Map | Return trung bình | Số nạn nhân cứu trung bình |
|---|---|---|
| map1 | ≈ −126 | ≈ 0,00 / 2 |
| map2 | ≈ −185 | ≈ 0,01 / 3 |
| map3 | ≈ −612 | ≈ 0,07 / 4 |

Policy ngẫu nhiên gần như không cứu được ai, nên là mốc dưới để đánh giá Q-learning và Monte Carlo. Kết quả huấn luyện của hai thuật toán nằm trong `results/` và được vẽ bằng `plot_results.py`.