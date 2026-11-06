# Đề tài 30 – Rescue Mission nhiều nạn nhân
## Phần 1: Mô hình MDP, môi trường và thiết kế map

### 1. Mô hình hóa MDP

**Môi trường**: lưới H×W, agent xuất phát ở `S`, cứu nạn nhân `V` và đưa về điểm an toàn `G`. Mỗi lần chỉ mang được 1 người. Một số nạn nhân cần medkit, medkit hữu hạn, ô `M` bổ sung đúng 1 medkit (dùng một lần). Ô `D` là ô nguy hiểm, `#` là tường.

**State**: `s = (r, c, mask, carry, med, supply_used)`

| Biến | Miền giá trị | Ý nghĩa |
|---|---|---|
| (r, c) | H×W | Vị trí agent |
| mask | 0 … 2^N − 1 | Bit i = 1 nếu nạn nhân i đã được đưa về G |
| carry | 0 … N | 0: không mang ai; i+1: đang mang nạn nhân i |
| med | 0 … K_max | Số medkit còn lại |
| supply_used | {0, 1} | Ô supply đã được dùng chưa |

Số state tối đa: `|S| = H·W·2^N·(N+1)·(K_max+1)·2`, với `K_max = medkits ban đầu + 1 (nếu có ô M)`.

**Action** (6 action rời rạc): Lên, Xuống, Trái, Phải, Pickup, Drop.

- *Pickup* hợp lệ khi agent đứng trên nạn nhân i chưa cứu, đang không mang ai, và (nạn nhân i không cần medkit hoặc `med > 0`). Hiệu ứng: `carry = i+1`; nếu nạn nhân cần medkit thì `med ← med − 1`.
- *Drop* hợp lệ khi agent đứng ở `G` và `carry > 0`. Hiệu ứng: đặt bit tương ứng trong `mask`, `carry = 0`.
- Bước vào ô `M` lần đầu: `med ← med + 1`, `supply_used = 1` (tự động).
- Hành động không hợp lệ (đâm tường, Pickup/Drop sai chỗ): agent đứng yên, bị phạt.

**Transition**: xác định (deterministic), `s' = f(s, a)`.

**Reward**

| Sự kiện | Reward |
|---|---|
| Mỗi bước | −1 |
| Hành động không hợp lệ | −5 |
| Bước vào ô D | −10 |
| Drop nạn nhân i về G | `+R_i` |
| Cứu đủ N nạn nhân | +100 |

Hai chế độ reward tạo hai policy cạnh tranh:
- **Mode A (cứu gần trước)**: `R_i = 50` cho mọi nạn nhân.
- **Mode B (ưu tiên giá trị cao)**: `R_i = 50 · value_i`.

**Terminal / truncation**
- *Terminated*: `mask = 2^N − 1` (cứu hết).
- *Truncated*: hết `max_steps`. Ô D chỉ bị phạt, không kết thúc episode.

**Discount**: γ = 0,99.

### 2. Tính Markov

State chứa đủ thông tin để xác định phân phối của trạng thái kế tiếp và reward:
- `mask` cho biết ai đã được cứu, nên không cần nhớ lịch sử đường đi.
- `carry` và `med` quyết định Pickup/Drop có hợp lệ và reward tiếp theo.
- `supply_used` đảm bảo ô supply chỉ bổ sung đúng 1 lần.

Do đó `P(s' | s, a, lịch sử) = P(s' | s, a)`. Giới hạn thời gian `max_steps` được xử lý như *truncation* (không đưa bước thời gian vào state), nên không phá tính Markov của MDP gốc.

### 3. Ba map/cấu hình tăng dần độ khó

| Map | Lưới | N | Medkit đầu | Nạn nhân cần medkit | Ô M | \|S\| | \|A\| | max_steps | Số bước tối thiểu |
|---|---|---|---|---|---|---|---|---|---|
| map1 | 5×5 | 2 | – | 0 | không | 600 | 6 | 30 | 24 |
| map2 | 6×6 | 3 | 1 | 2 | có | 6.912 | 6 | 45 | 33 |
| map3 | 7×7 | 4 | 2 | 3 | có | 31.360 | 6 | 140 | 105 |

Ánh xạ với các setting bắt buộc của đề:

| Setting của đề | Map |
|---|---|
| 1. 2 nạn nhân, không giới hạn nguồn lực | map1 |
| 2. 3–4 nạn nhân, mang 1 người/lần | map2, map3 |
| 3. Một số nạn nhân cần medkit, medkit hữu hạn | map2, map3 |
| 4. Ô supply bổ sung đúng 1 medkit, phải đi đường vòng | map2 (31 → 33 bước, +2); map3 có ô supply nhưng nằm trên đường tối ưu (không đi vòng) |

Số bước tối thiểu được tính bằng BFS trên không gian state (đã gồm Pickup/Drop). Ở map3, ô `D` có đường né (mở tường ô (5,4)), nên đi qua `D` là lựa chọn của policy chứ không bắt buộc.

### 4. Policy tối ưu tham chiếu (Value Iteration, γ = 0,99)

| Map | Mode | Thứ tự cứu | Số bước | Return |
|---|---|---|---|---|
| map1 | A | V0 → V1 | 24 | 176 |
| map1 | B | V0 → V1 | 24 | 276 |
| map2 | A | V2 → V1 → V0 | 33 | 217 |
| map2 | B | V0 → V1 → V2 | 33 | 367 |
| map3 | A | V1 → V2 → V3 → V0 | 109 | 191 |
| map3 | B | V1 → V0 → V2 → V3 | 109 | 391 |

Ở map2 và map3, hai mode cho thứ tự cứu khác nhau (A ưu tiên nạn nhân gần, B ưu tiên giá trị cao). Vì cả hai cứu đủ trong cùng số bước nên return chưa chiết khấu bằng nhau giữa các thứ tự; sự khác biệt chỉ thể hiện qua chiết khấu γ và thời điểm giao từng nạn nhân.

### 5. Baseline ngẫu nhiên (200 episode)

| Map | Return trung bình | Số nạn nhân cứu trung bình |
|---|---|---|
| map1 | ≈ −126 | ≈ 0,00 / 2 |
| map2 | ≈ −185 | ≈ 0,01 / 3 |
| map3 | ≈ −612 | ≈ 0,07 / 4 |

Policy ngẫu nhiên gần như không cứu được ai, nên là mốc dưới để đánh giá Q-learning và Monte Carlo.
