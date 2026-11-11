"""阶段 3：抗延迟的学生策略——把动作延迟加进采集和训练，目标是延迟上限超过老师的约 40 ms。

三件事按计划定下来：
  1. 观测里的"上一帧动作"填**学生自己上一步的输出**（和老师语义一致，采集时天然如此）；
  2. 把本回合的延迟 d 作为**额外输入**（1083 → 1084 维，最后一维 = d_ms / 60）；
  3. **标签前移 d**：学生走完一整回合记录全部观测，事后用老师在 t+k 步观测上的动作作为第 t 步的标签
     （k = d/10，策略步数；回合末尾 k 步丢掉）。`--no-shift` 走对照组：标签取老师在同一时刻的动作。

为什么这样能补偿延迟：学生会学到"在延迟 d 下、t 时刻该发出、直到 t+k 才真正被执行的动作"，
相当于提前把动作往前打，抵消电机执行的滞后。

评测：
  - 延迟 0~80 ms 每 10 ms 一档（70/80 ms 超出训练范围，单独标注）；
  - d = 0 时的速度跟踪（vx 0.3~0.6）和抗推（4 个方向），确认没有为抗延迟牺牲原有能力。

用法：
  python 30_t800_delay_student.py --rounds 1 --steps 6000 --epochs 3 --eval-latency 0,20   # 小规模试跑
  python 30_t800_delay_student.py                                                          # 完整版
  python 30_t800_delay_student.py --no-shift                                                # 对照组

输出：
  data/delay_r{k}.npz / .csv     第 k 轮采集的（学生状态 + 前移后的老师标签 + 该样本的 d）
  models/delay_mlp_r{k}.pt       第 k 轮的学生
  results/delay_latency.csv      每轮的延迟曲线（含老师对照）
  results/delay_final.csv        d=0 时的速度跟踪与抗推（和上一阶段的学生对比）
"""
import argparse
import csv
import ctypes
import importlib.util
import time
import traceback
import winsound
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def notify(title, msg, error=False):
    """跑完 / 出错时弹一个置顶窗口（不装新库，用系统自带 MessageBox）。"""
    winsound.MessageBeep(winsound.MB_ICONHAND if error else winsound.MB_ICONASTERISK)
    # 0x10 错误图标 / 0x40 信息图标，0x40000 置顶，0x10000 设为前台
    ctypes.windll.user32.MessageBoxW(0, msg, title, (0x10 if error else 0x40) | 0x40000 | 0x10000)

CODE_DIR = Path(__file__).resolve().parent
DATA_DIR = CODE_DIR / "data"
RESULT_DIR = CODE_DIR / "results"
MODEL_DIR = CODE_DIR / "models"

# ---------- 设置 ----------
ROUNDS = 3
STEPS_PER_ROUND = 60000
DURATION = 15.0
D_TRAIN_MS = (0, 10, 20, 30, 40, 50, 60)   # 训练时的延迟档位（10 ms 整数倍，和第 3 节的说明一致）
D_EVAL_MS = tuple(range(0, 81, 10))        # 评测到 80 ms，70/80 超出训练范围
D_NORM = 60.0                              # 输入里的 d = d_ms / D_NORM
EPOCHS = 30
BATCH = 1024
LR = 1e-3
VAL_RATIO = 0.1
SEED = 0
DELAY_FRACTION_TARGET = 1 / 3              # 训练集里 d > 0 的样本占比目标
BASE_DATA = ("data/bc_teacher.npz", "data/dagger_r1.npz", "data/dagger_r2.npz",
             "data/dagger_r3.npz", "data/dagger_r4.npz")   # 阶段 1、2 的数据，d 一律按 0 处理
VX_RANGE = (0.3, 0.6)
INIT_NOISE = 0.10
PUSH_TIME_RANGE = (3.0, 12.0)
PUSH_FORCE_RANGE = (40.0, 120.0)
PUSH_DURATION = 0.2
MAX_PUSHES = 3
EVAL_SEEDS = range(10)
SPEED_CMDS = (0.3, 0.4, 0.5, 0.6)
PUSH_DIRS = {"向前 (+x)": (1, 0), "向后 (-x)": (-1, 0), "向左 (+y)": (0, 1), "向右 (-y)": (0, -1)}
PUSH_FORCES = np.arange(50, 801, 25)
PASS_RATE = 0.8

spec = importlib.util.spec_from_file_location("t27", CODE_DIR / "27_t800_bc_train.py")
t27 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t27)
r25 = t27.load_r25()
H = r25.HISTORY


def student_fn(net, d_ms):
    """把网络包成 run_trial / 采集需要的接口：观测 1083 维 + 该时刻的延迟 d。

    阶段 2 的学生是 1083 维输入（不知道 d），这里自动兼容：输入维度是 1083 就直接喂。
    """
    net.eval()
    in_dim = net[0].in_features
    d = np.float32(d_ms / D_NORM)

    def policy(obs):
        x = np.asarray(obs, np.float32)
        if in_dim > x.size:
            x = np.concatenate([x, [d]])
        with torch.no_grad():
            return net(torch.from_numpy(x)[None]).numpy()[0].astype(np.float32)

    return policy


def build_obs_at(frames, cmds, j):
    """重建某回合第 j 步的 1083 维观测（15 帧历史，前 14 步用第一帧补），用于事后打标签。"""
    idx = np.clip(j - H + 1 + np.arange(H), 0, j)
    return np.concatenate([frames[idx].reshape(-1), cmds[j]])


def save_round(path, recs, infos):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, frames=np.concatenate([r["frames"] for r in recs]),
             actions=np.concatenate([r["actions"] for r in recs]),
             cmds=np.concatenate([r["cmds"] for r in recs]),
             delay_ms=np.concatenate([r["delay_ms"] for r in recs]),
             episode=np.concatenate([np.full(len(r["frames"]), i, np.int32) for i, r in enumerate(recs)]),
             step=np.concatenate([np.arange(len(r["frames"]), dtype=np.int32) for r in recs]))
    with open(path.with_suffix(".csv"), "w", encoding="utf-8-sig", newline="") as f:
        f.write("回合,种子,延迟ms,指令vx,是否摔倒,记录步数,结束时刻(s)\n")
        for i, info in enumerate(infos):
            f.write(f"{i},{info['seed']},{info['d_ms']},{info['vx']:.3f},{'摔倒' if info['fell'] else '没摔'},"
                    f"{info['steps']},{info['t_end']:.3f}\n")


def load_part(path):
    """读一份数据；老数据（阶段 1、2）没有 delay_ms 字段，按 0 处理。"""
    d = np.load(path)
    delay = d["delay_ms"] if "delay_ms" in d.files else np.zeros(len(d["frames"]), np.int32)
    return d["frames"], d["actions"], d["cmds"], d["episode"], d["step"], delay


def load_all(paths):
    """返回 (frames, actions, cmds, delays, episode, step)，顺序和 train_model 的参数一致。"""
    parts = [load_part(p) for p in paths]
    offsets = np.cumsum([0] + [int(p[3].max()) + 1 for p in parts[:-1]])
    return (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]),
            np.concatenate([p[2] for p in parts]),
            np.concatenate([p[5] for p in parts]).astype(np.int32),
            np.concatenate([p[3] + off for p, off in zip(parts, offsets)]).astype(np.int32),
            np.concatenate([p[4] for p in parts]))


def make_obs_np(frames, cmds, delays, idx):
    """(B,15) 历史下标 → (B,1084)：1080 帧 + 3 指令 + 1 延迟。"""
    cur = idx[:, -1]
    return np.concatenate([frames[idx].reshape(len(idx), -1), cmds[cur],
                           (delays[cur] / D_NORM)[:, None].astype(np.float32)], axis=1)


def train_model(frames, actions, cmds, delays, episode, step, epochs, tag, seed=SEED):
    """按回合划分训练 / 验证；训练前先报告 d>0 的占比，不足目标占比就过采样带延迟的样本。"""
    n_ep = int(episode.max()) + 1
    idx_all = t27.history_index(episode, step, H)
    rng = np.random.default_rng(seed)
    n_val = max(1, round(n_ep * VAL_RATIO))
    is_val = np.isin(episode, rng.permutation(n_ep)[:n_val])
    train_rows, val_rows = np.flatnonzero(~is_val), np.flatnonzero(is_val)

    delayed = train_rows[delays[train_rows] > 0]
    frac = len(delayed) / max(1, len(train_rows))
    print(f"  [{tag}] 训练集 {len(train_rows)} 步，其中 d>0 的 {len(delayed)} 步 = {frac:.1%}", flush=True)
    rows = train_rows
    if frac < DELAY_FRACTION_TARGET and len(delayed):
        r = int(np.ceil((len(train_rows) - len(delayed)) / (2.0 * len(delayed))))
        rows = np.concatenate([train_rows] + [delayed] * (r - 1))
        print(f"  [{tag}] d>0 占比低于 {DELAY_FRACTION_TARGET:.0%}，采取**过采样**（带延迟样本重复 {r} 次），"
              f"过采样后 {len(delayed) * r / len(rows):.1%}（d=0 样本一条不丢）", flush=True)

    torch.manual_seed(seed)
    net = t27.build_net(H * frames.shape[1] + cmds.shape[1] + 1, actions.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    best, t0 = np.inf, time.time()
    for epoch in range(1, epochs + 1):
        net.train()
        perm = rng.permutation(rows)
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            loss = loss_fn(net(torch.from_numpy(make_obs_np(frames, cmds, delays, idx_all[b]))),
                           torch.from_numpy(actions[b]))
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        va = batch_val(net, loss_fn, frames, cmds, delays, actions, val_rows, idx_all)
        best = min(best, va)
        if epoch % max(1, epochs // 6) == 0 or epoch == epochs:
            print(f"    [{tag}] 第 {epoch}/{epochs} 轮  验证 MSE {va:.5f}（最好 {best:.5f}）", flush=True)
    return net, best, time.time() - t0


def batch_val(net, loss_fn, frames, cmds, delays, actions, rows, idx_all, batch=2048):
    total = 0.0
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            b = rows[i:i + batch]
            total += loss_fn(net(torch.from_numpy(make_obs_np(frames, cmds, delays, idx_all[b]))),
                             torch.from_numpy(actions[b])).item() * len(b)
    return total / len(rows)


def collect_round(net, steps_target, seed0, shift=True):
    """学生走（动作经过 d 步延迟执行），记录观测；回合结束后用老师在这个回合 t+k 步的观测打标签。"""
    model, data = r25.model, r25.data
    recs, infos, done, ep = [], [], 0, 0
    while done < steps_target:
        rng = np.random.default_rng(seed0 + ep)
        d_ms = int(rng.choice(D_TRAIN_MS))
        vx = float(rng.uniform(*VX_RANGE))
        pushes = []
        for _ in range(int(rng.integers(0, MAX_PUSHES + 1))):
            ang = rng.uniform(0, 2 * np.pi)
            mag = rng.uniform(*PUSH_FORCE_RANGE)
            pushes.append((float(rng.uniform(*PUSH_TIME_RANGE)), mag * np.cos(ang), mag * np.sin(ang)))
        delay = round(d_ms / 1000 / r25.DT)          # 延迟换算成仿真步
        policy = student_fn(net, d_ms)

        r25.mujoco.mj_resetData(model, data)
        data.qpos[r25.qadr] = r25.DEFAULT_Q
        data.qpos[r25.qadr[r25.POLICY]] += rng.uniform(-INIT_NOISE, INIT_NOISE, len(r25.POLICY))
        r25.mujoco.mj_forward(model, data)

        history, action = None, np.zeros(22, np.float32)
        q_des = r25.DEFAULT_Q.copy()
        q_buf = deque([r25.DEFAULT_Q.copy()] * (delay + 1), maxlen=delay + 1)  # 执行的是 delay 步之前的目标角
        desired_yaw = r25.yaw_of(data.qpos[3:7])
        frames, cmds = [], []
        fell = False
        for step in range(round(DURATION / r25.DT)):
            if step % r25.DECIMATION == 0:
                R = data.xmat[r25.base_id].reshape(3, 3)
                gravity = R.T @ [0, 0, -1]
                if data.qpos[2] < r25.FALL_HEIGHT or gravity[2] > -np.cos(np.radians(r25.FALL_TILT)):
                    fell = True
                    break
                q, dq = data.qpos[r25.qadr], data.qvel[r25.dadr]
                frame = np.clip(np.concatenate([(q - r25.DEFAULT_Q)[r25.POLICY], dq[r25.POLICY] * r25.DQ_SCALE,
                                                action, data.qvel[3:6], gravity]), -r25.CLIP, r25.CLIP).astype(np.float32)
                history = [frame] * H if history is None else history[1:] + [frame]
                yaw_err = np.angle(np.exp(1j * (desired_yaw - r25.yaw_of(data.qpos[3:7]))))
                cmd = np.array([vx * r25.CMD_SCALE, 0.0, np.clip(r25.YAW_KP * yaw_err, -1, 1)], np.float32)
                action = np.clip(policy(np.concatenate(history + [cmd])), -r25.CLIP, r25.CLIP)
                q_des = r25.DEFAULT_Q.copy()
                q_des[r25.POLICY] += action * r25.ACTION_SCALE
                frames.append(frame)
                cmds.append(cmd)

            q_buf.append(q_des)
            tau = r25.KP * (q_buf[0] - data.qpos[r25.qadr]) - r25.KD * data.qvel[r25.dadr]
            data.ctrl[:] = np.clip(tau, -r25.tau_limit, r25.tau_limit)
            fx = fy = 0.0
            for t0, px, py in pushes:
                if t0 <= data.time < t0 + PUSH_DURATION:
                    fx, fy = px, py
            data.xfrc_applied[r25.base_id, :2] = (fx, fy)
            r25.mujoco.mj_step(model, data)

        # ---------- 事后打标签：第 t 步的目标 = 老师在 t+k 步观测上的动作 ----------
        frames, cmds = np.array(frames, np.float32), np.array(cmds, np.float32)
        T = len(frames)
        k = d_ms // 10 if shift else 0     # 策略步为单位的延迟（10 ms 一步）
        keep = T - k
        labels = np.array([r25.infer(build_obs_at(frames, cmds, t + k)) for t in range(keep)], np.float32) \
            if keep > 0 else np.zeros((0, 22), np.float32)
        if keep > 0:
            recs.append(dict(frames=frames[:keep], actions=labels, cmds=cmds[:keep],
                             delay_ms=np.full(keep, d_ms, np.int32)))
            done += keep
        infos.append(dict(seed=seed0 + ep, vx=vx, d_ms=d_ms, steps=max(keep, 0), fell=fell,
                          t_end=float(data.time)))
        ep += 1
        if ep >= 3000:  # 兜底：万一每个回合都极短（比如一上来就摔），避免死循环
            print("  警告：尝试次数过多，提前结束本轮回采集", flush=True)
            break
    return recs, infos, done


# ---------- 评测 ----------
def trial(policy, d_ms, vx=0.5):
    res = [r25.run_trial(vx, 0.0, seed=s, delay_ms=d_ms, policy=policy) for s in EVAL_SEEDS]
    return float(r25.fall_rate(res)), float(r25.mean_speed(res, 2))


def read_csv(path):
    """读 CSV（给报告里做 "上一阶段 vs 本阶段" 的对比用）。"""
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def eval_latency(net, tag, delays):
    """学生在这几个延迟档位下的摔倒率 / 实际速度。"""
    out, policy_cache = [], {}
    for d in delays:
        policy_cache[d] = student_fn(net, d)
        fr, avx = trial(policy_cache[d], d)
        out.append([d, f"学生_{tag}", round(fr, 3), round(avx, 3) if not np.isnan(avx) else "",
                    "是" if d > max(D_TRAIN_MS) else "否"])
        print(f"    延迟 {d:>2} ms：学生摔倒率 {fr:>4.0%}，实际 vx {avx:.3f}", flush=True)
    return out


def eval_speed(net):
    rows = []
    for vx in SPEED_CMDS:
        fr, avx = trial(student_fn(net, 0), 0, vx=vx)
        rows.append([vx, "阶段3学生", round(avx, 3) if not np.isnan(avx) else "",
                     round(abs(avx - vx), 3) if not np.isnan(avx) else "", round(fr, 3)])
        print(f"    指令 vx {vx:.1f}：阶段3学生 {avx:.3f}（误差 {abs(avx - vx):.3f}），摔倒率 {fr:.0%}", flush=True)
    return rows


def eval_push(net):
    """四个方向逐档加力，连续 2 档全摔就停（和 25 的抗推实验一致）。"""
    rows, policy = [], student_fn(net, 0)
    for name, (ux, uy) in PUSH_DIRS.items():
        print(f"    {name}：", flush=True)
        zero = 0
        for f in PUSH_FORCES:
            if zero >= 2:
                break
            res = [r25.run_trial(0.5, 0.0, seed=s, push=(f * ux, f * uy), policy=policy) for s in EVAL_SEEDS]
            fr = float(r25.fall_rate(res))
            rows.append([name, int(f), "阶段3学生", round(fr, 3)])
            zero = zero + 1 if fr == 1.0 else 0
            print(f"      {f:>4} N  摔倒率 {fr:>4.0%}", flush=True)
    return rows


def write_rows(path, header, rows):
    t27.write_csv(CODE_DIR / path, header, rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rounds", type=int, default=ROUNDS, help=f"轮数，默认 {ROUNDS}")
    parser.add_argument("--steps", type=int, default=STEPS_PER_ROUND, help=f"每轮采集步数，默认 {STEPS_PER_ROUND}")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help=f"每轮训练轮数，默认 {EPOCHS}")
    parser.add_argument("--seed0", type=int, default=0)
    parser.add_argument("--no-shift", action="store_true", help="对照组：标签不前移 d")
    parser.add_argument("--eval-latency", default="", help="只评测这几个延迟档位，逗号分隔（试跑用）")
    parser.add_argument("--skip-final", action="store_true", help="跳过速度跟踪和抗推（试跑用）")
    parser.add_argument("--driver", default="models/bc_mlp_r4.pt", help="第 1 轮采集时用哪个策略来驱动")
    parser.add_argument("--train-seed", type=int, default=SEED, help="训练用的随机种子（默认 0），用于评估结果波动")
    parser.add_argument("--train-only", action="store_true", help="不采集，直接用已有的 delay_r{k}.npz 重新训练")
    parser.add_argument("--out-suffix", default="", help="结果 CSV / 模型名的后缀，避免互相覆盖")
    parser.add_argument("--eval-only", default="", help="只对已有模型测 d=0 的速度跟踪和抗推")
    args = parser.parse_args()

    shift = not args.no_shift
    sfx = args.out_suffix
    if args.eval_only:  # 只评测：速度跟踪 + 抗推（都是 d = 0）
        net = t27.load_student(args.eval_only)[0]
        print(f"仅评测 {args.eval_only}", flush=True)
        print("d=0 速度跟踪", flush=True)
        speed_rows = eval_speed(net)
        write_rows(f"results/delay_speed{sfx}.csv", ["指令vx", "策略", "实际vx", "跟踪误差", "摔倒率"], speed_rows)
        print("d=0 抗推", flush=True)
        push_rows = eval_push(net)
        write_rows(f"results/delay_push{sfx}.csv", ["方向", "推力(N)", "策略", "摔倒率"], push_rows)
        summary = f"仅评测 {args.eval_only}：d=0 速度跟踪与抗推完成"
        print(summary, flush=True)
        return summary
    delays = [int(d) for d in args.eval_latency.split(",")] if args.eval_latency else list(D_EVAL_MS)
    print(f"标签前移：{'是（主方案）' if shift else '否（对照组）'}；训练延迟档位 {D_TRAIN_MS} ms；"
          f"评测延迟 {delays} ms", flush=True)

    paths = [CODE_DIR / p for p in BASE_DATA]
    net = t27.load_student(args.driver)[0]   # 第 1 轮用阶段 2 的学生驱动（1083 维，不知道 d）
    lat_rows = []
    for k in range(1, args.rounds + 1):
        t0 = time.time()
        if args.train_only:  # 只重训练已有数据（用于换训练种子重复实验）
            out_npz = DATA_DIR / f"delay_r{k}{'' if shift else '_noshift'}.npz"
            print(f"第 {k} 轮：跳过采集，直接用 {out_npz.name}", flush=True)
        else:
            recs, infos, done = collect_round(net, args.steps, args.seed0 + 1000 * k, shift)
            out_npz = DATA_DIR / f"delay_r{k}{'' if shift else '_noshift'}.npz"
            save_round(out_npz, recs, infos)
            by_d = {}
            for i in infos:
                by_d.setdefault(i["d_ms"], [0, 0])
                by_d[i["d_ms"]][0] += 1
                by_d[i["d_ms"]][1] += i["fell"]
            print(f"第 {k} 轮采集：{done} 步 / {len(infos)} 次尝试，用时 {time.time() - t0:.0f} s；"
                  f"各档（尝试次数/摔倒次数）：{ {d: tuple(v) for d, v in sorted(by_d.items())} }", flush=True)
        paths.append(out_npz)

        data = load_all(paths)
        net, best, dt = train_model(*data, args.epochs, f"r{k}", seed=args.train_seed)
        seed_tag = f"_s{args.train_seed}" if args.train_seed else ""
        torch.save(dict(state_dict=net.state_dict(), in_dim=H * 72 + 3 + 1, out_dim=22, hidden=t27.HIDDEN,
                        history=H, epoch=args.epochs, val_loss=best, train_seed=args.train_seed,
                        data=f"base + delay_r1..{k}{'' if shift else '_noshift'}"),
                   MODEL_DIR / f"delay_mlp_r{k}{'' if shift else '_noshift'}{seed_tag}{sfx}.pt")
        print(f"  第 {k} 轮训练完成（{dt:.0f} s，验证 MSE {best:.5f}），开始评测", flush=True)
        lat_rows += eval_latency(net, f"r{k}", delays)
        write_rows(f"results/delay_latency{sfx}.csv", ["延迟ms", "策略", "摔倒率", "实际vx", "超出训练范围(>60ms)"], lat_rows)

    if not args.skip_final:
        print("最终模型：d=0 速度跟踪", flush=True)
        speed_rows = eval_speed(net)
        write_rows(f"results/delay_speed{sfx}.csv", ["指令vx", "策略", "实际vx", "跟踪误差", "摔倒率"], speed_rows)
        print("最终模型：d=0 抗推", flush=True)
        push_rows = eval_push(net)
        write_rows(f"results/delay_push{sfx}.csv", ["方向", "推力(N)", "策略", "摔倒率"], push_rows)

    # 小结：主方案 / 对照组 + 最后一轮的 0 ms 与 40 ms 摔倒率（弹窗内容）
    tag = f"学生_r{args.rounds}"

    def rate(d):
        v = [r[2] for r in lat_rows if r[0] == d and r[1] == tag]
        return f"{v[0]:.0%}" if v else "未测"

    summary = (f"{'主方案（标签前移 d）' if shift else '对照组（标签不前移）'}，{args.rounds} 轮；"
               f"最后一轮延迟 0 ms 摔倒率 {rate(0)}，40 ms 摔倒率 {rate(40)}")
    print(summary, flush=True)
    return summary


if __name__ == "__main__":
    try:
        summary = main()
    except Exception:
        last = traceback.format_exc().strip().splitlines()[-1]   # 异常最后一行，弹窗里显示
        notify("T800 任务出错", last, error=True)
        raise                                                    # 让日志里仍有完整报错
    else:
        notify("T800 任务完成", summary)


