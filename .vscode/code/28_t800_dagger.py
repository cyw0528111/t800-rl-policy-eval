"""DAgger：让学生自己走、用老师的动作打标签，把学生走到的状态补进数据集重新训练（阶段 2）。

为什么需要它：阶段 1 的行为克隆在老师走过的状态上动作误差只有 0.024（老师动作 std 1.2 的 2%），
但一闭环就 4.6~7.0 s 必摔。诊断（_diag_bc.py）显示"老师此刻会给的动作"与"学生的动作"的偏差
会从走路阶段的约 1.0 一路涨到摔倒前的 10~18，典型的复合误差：
学生一旦偏离，它自己的动作还会进入观测历史（"上一帧动作"），状态越偏越远。

DAgger 的做法：
  1. 让学生自己控制（它才会走到自己会犯错的状态），每一步把这些状态交给老师打标签；
  2. 把"学生的状态 + 老师的动作"并进数据集，重新训练学生；
  3. 重复若干轮，每轮评测一次。

摔倒了就重置继续采集，直到凑够本轮的步数预算，这样摔倒前后的状态都能被覆盖到。

用法：
  python 28_t800_dagger.py --rounds 1 --steps 6000 --epochs 3    # 小规模试跑，确认流程
  python 28_t800_dagger.py --rounds 4 --steps 60000 --epochs 30  # 完整版（默认）

输出：
  data/dagger_r{k}.npz / .csv    第 k 轮采集的（学生状态，老师标签）
  models/bc_mlp_r{k}.pt          第 k 轮训练出的学生（第 0 轮是阶段 1 的纯行为克隆，不重训）
  results/dagger_rounds.csv      每轮的采集步数、验证损失、各指令摔倒率
  results/dagger_curve.png       摔倒率 / 成功率随轮次的变化
"""
import argparse
import csv
import importlib.util
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE_DIR = Path(__file__).resolve().parent
DATA_DIR = CODE_DIR / "data"
RESULT_DIR = CODE_DIR / "results"
MODEL_DIR = CODE_DIR / "models"

# ---------- 设置 ----------
ROUNDS = 4                    # DAgger 轮数
STEPS_PER_ROUND = 60000       # 每轮采集的步数（100 Hz 下 60000 步 = 600 s 的"学生走"数据）
DURATION = 15.0               # 单次仿真最长时长 (s)，摔了就提前结束并重置
EPOCHS = 30                   # 每轮重训练的轮数
BATCH = 1024
LR = 1e-3
VAL_RATIO = 0.1
SEED = 0
BASE_DATA = "data/bc_teacher.npz"   # 阶段 1 采集的老师数据，始终参与训练
BC_MODEL = "models/bc_mlp.pt"       # 阶段 1 的纯行为克隆权重，作为第 0 轮
VX_RANGE = (0.3, 0.6)
INIT_NOISE = 0.10             # 和 26 的采集设置保持一致
PUSH_TIME_RANGE = (3.0, 12.0)
PUSH_FORCE_RANGE = (40.0, 120.0)
PUSH_DURATION = 0.2
MAX_PUSHES = 3
EVAL_CMDS = (0.3, 0.4, 0.5, 0.6)
EVAL_SEEDS = range(5)

# 复用 27 里的网络结构、观测拼装和 25 的加载函数，避免重写一份
spec = importlib.util.spec_from_file_location("t27", CODE_DIR / "27_t800_bc_train.py")
t27 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t27)
r25 = t27.load_r25()
H = r25.HISTORY


def student_fn(net):
    """把网络包成 run_trial(policy=...) 需要的 numpy 进 / numpy 出。"""
    net.eval()

    def policy(obs):
        with torch.no_grad():
            return net(torch.from_numpy(np.asarray(obs, np.float32))[None]).numpy()[0].astype(np.float32)

    return policy


def collect_round(student, steps_target, seed0):
    """让学生走，记录每一步的 72 维帧和老师的动作标签；摔倒就重置，直到凑够 steps_target 步。"""
    model, data = r25.model, r25.data
    recs, infos, done, ep = [], [], 0, 0
    while done < steps_target:
        rng = np.random.default_rng(seed0 + ep)
        vx = float(rng.uniform(*VX_RANGE))
        pushes = []
        for _ in range(int(rng.integers(0, MAX_PUSHES + 1))):
            ang = rng.uniform(0, 2 * np.pi)
            mag = rng.uniform(*PUSH_FORCE_RANGE)
            pushes.append((float(rng.uniform(*PUSH_TIME_RANGE)), mag * np.cos(ang), mag * np.sin(ang)))

        r25.mujoco.mj_resetData(model, data)
        data.qpos[r25.qadr] = r25.DEFAULT_Q
        data.qpos[r25.qadr[r25.POLICY]] += rng.uniform(-INIT_NOISE, INIT_NOISE, len(r25.POLICY))
        r25.mujoco.mj_forward(model, data)

        history, action = None, np.zeros(22, np.float32)
        q_des = r25.DEFAULT_Q.copy()
        desired_yaw = r25.yaw_of(data.qpos[3:7])
        frames, labels, cmds = [], [], []
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
                obs = np.concatenate(history + [cmd])
                label = np.clip(r25.infer(obs), -r25.CLIP, r25.CLIP)     # 老师的标签
                action = np.clip(student(obs), -r25.CLIP, r25.CLIP)      # 学生真正执行的动作，会进入下一帧历史
                q_des = r25.DEFAULT_Q.copy()
                q_des[r25.POLICY] += action * r25.ACTION_SCALE
                frames.append(frame)
                labels.append(label)
                cmds.append(cmd)

            tau = r25.KP * (q_des - data.qpos[r25.qadr]) - r25.KD * data.qvel[r25.dadr]
            data.ctrl[:] = np.clip(tau, -r25.tau_limit, r25.tau_limit)
            fx = fy = 0.0
            for t0, px, py in pushes:
                if t0 <= data.time < t0 + PUSH_DURATION:
                    fx, fy = px, py
            data.xfrc_applied[r25.base_id, :2] = (fx, fy)
            r25.mujoco.mj_step(model, data)

        recs.append(dict(frames=np.array(frames, np.float32), actions=np.array(labels, np.float32),
                         cmds=np.array(cmds, np.float32)))
        done += len(frames)
        infos.append(dict(seed=seed0 + ep, vx=vx, steps=len(frames), fell=fell, t_end=float(data.time)))
        ep += 1
    return recs, infos, done


def save_round(path, recs, infos):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, frames=np.concatenate([r["frames"] for r in recs]),
             actions=np.concatenate([r["actions"] for r in recs]),
             cmds=np.concatenate([r["cmds"] for r in recs]),
             episode=np.concatenate([np.full(len(r["frames"]), i, np.int32) for i, r in enumerate(recs)]),
             step=np.concatenate([np.arange(len(r["frames"]), dtype=np.int32) for r in recs]))
    with open(path.with_suffix(".csv"), "w", encoding="utf-8-sig", newline="") as f:
        f.write("回合,种子,指令vx,是否摔倒,记录步数,结束时刻(s)\n")
        for i, info in enumerate(infos):
            f.write(f"{i},{info['seed']},{info['vx']:.3f},{'摔倒' if info['fell'] else '没摔'},"
                    f"{info['steps']},{info['t_end']:.3f}\n")


def load_all(paths):
    """把基础数据集和各轮 DAgger 数据拼起来，回合编号重新连续编。"""
    parts = [t27.load_data(p) for p in paths]
    offsets = np.cumsum([0] + [int(p["episode"].max()) + 1 for p in parts[:-1]])
    return (np.concatenate([p["frames"] for p in parts]), np.concatenate([p["actions"] for p in parts]),
            np.concatenate([p["cmds"] for p in parts]),
            np.concatenate([p["episode"] + off for p, off in zip(parts, offsets)]).astype(np.int32),
            np.concatenate([p["step"] for p in parts]))


def load_net(path):
    ckpt = torch.load(CODE_DIR / path, map_location="cpu", weights_only=False)
    net = t27.build_net(ckpt["in_dim"], ckpt["out_dim"], ckpt["hidden"])
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, ckpt


def evaluate_round(net, seeds=EVAL_SEEDS):
    """用 25 的 run_trial 评测：每个指令跑 5 个种子，返回 {vx: (摔倒率, 实际 vx)}。"""
    policy = student_fn(net)
    out = {}
    for vx in EVAL_CMDS:
        res = [r25.run_trial(vx, 0.0, seed=s, policy=policy) for s in seeds]
        out[vx] = (float(r25.fall_rate(res)), float(r25.mean_speed(res, 2)))
    return out


def plot_rounds(rows):
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    rounds = [r[0] for r in rows]
    for k, vx in enumerate(EVAL_CMDS):
        axes[0].plot(rounds, [100 * (1 - r[3 + k]) for r in rows], "-o", color=colors[k],
                     linewidth=2, markersize=6, label=f"vx={vx}")
    axes[0].set(xlabel="DAgger 轮次（0 = 纯行为克隆）", ylabel="15 s 行走成功率 (%)", ylim=(-5, 105),
                title="各指令的行走成功率")
    axes[1].plot(rounds, [100 * (1 - r[7]) for r in rows], "-o", color="#2a78d6", linewidth=2, markersize=7)
    axes[1].set(xlabel="DAgger 轮次", ylabel="平均成功率 (%)", ylim=(-5, 105), title="四个指令平均")
    for ax in axes:
        ax.grid(alpha=0.3, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "dagger_curve.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rounds", type=int, default=ROUNDS, help=f"DAgger 轮数，默认 {ROUNDS}")
    parser.add_argument("--steps", type=int, default=STEPS_PER_ROUND, help=f"每轮采集步数，默认 {STEPS_PER_ROUND}")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help=f"每轮重训练轮数，默认 {EPOCHS}")
    parser.add_argument("--seed0", type=int, default=0, help="采集用的起始种子")
    args = parser.parse_args()

    RESULT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    rows, paths = [], [CODE_DIR / BASE_DATA]

    # 第 0 轮：阶段 1 的纯行为克隆，作为对照
    net, ckpt = load_net(BC_MODEL)
    ev = evaluate_round(net)
    rows.append([0, 0, round(float(ckpt["val_loss"]), 6)] + [round(ev[vx][0], 3) for vx in EVAL_CMDS]
                + [round(np.mean([ev[vx][0] for vx in EVAL_CMDS]), 3)] + [round(ev[0.5][1], 3)])
    print(f"第 0 轮（纯行为克隆）：摔倒率 " + "，".join(f"vx={vx} {ev[vx][0]:.0%}" for vx in EVAL_CMDS), flush=True)

    for k in range(1, args.rounds + 1):
        t0 = time.time()
        recs, infos, done = collect_round(student_fn(net), args.steps, args.seed0 + 1000 * k)
        out_npz = DATA_DIR / f"dagger_r{k}.npz"
        save_round(out_npz, recs, infos)
        fell = sum(i["fell"] for i in infos)
        print(f"第 {k} 轮采集：{done} 步 / {len(infos)} 次尝试，其中摔倒 {fell} 次，"
              f"平均每次 {done / len(infos):.0f} 步，用时 {time.time() - t0:.0f} s → {out_npz.name}", flush=True)
        paths.append(out_npz)

        frames, actions, cmds, episode, step = load_all(paths)
        print(f"  重训练：{len(frames)} 步（基础 {len(t27.load_data(CODE_DIR / BASE_DATA)['frames'])} 步）", flush=True)
        net, best, dt = train_model(frames, actions, cmds, episode, step, args.epochs, f"r{k}")
        torch.save(dict(state_dict=net.state_dict(), in_dim=H * 72 + 3, out_dim=22, hidden=t27.HIDDEN,
                        history=H, epoch=args.epochs, val_loss=best, data=f"bc_teacher + dagger_r1..{k}"),
                   MODEL_DIR / f"bc_mlp_r{k}.pt")

        ev = evaluate_round(net)
        rows.append([k, done, round(best, 6)] + [round(ev[vx][0], 3) for vx in EVAL_CMDS]
                    + [round(np.mean([ev[vx][0] for vx in EVAL_CMDS]), 3)] + [round(ev[0.5][1], 3)])
        print(f"第 {k} 轮评测：摔倒率 " + "，".join(f"vx={vx} {ev[vx][0]:.0%}" for vx in EVAL_CMDS)
              + f"，平均 {np.mean([ev[vx][0] for vx in EVAL_CMDS]):.0%}（训练 {dt:.0f} s）", flush=True)

        t27.write_csv(RESULT_DIR / "dagger_rounds.csv",
                      ["轮次", "采集步数", "验证MSE"] + [f"摔倒率_vx{vx}" for vx in EVAL_CMDS]
                      + ["平均摔倒率", "实际vx_vx0.5"], rows)
        plot_rounds(rows)

    print("完成：results/dagger_rounds.csv, results/dagger_curve.png", flush=True)


def train_model(frames, actions, cmds, episode, step, epochs, tag):
    """在拼好的数据上重新训练，返回 (net, 最好验证 MSE, 用时)。按回合划分训练 / 验证。"""
    n_ep = int(episode.max()) + 1
    idx_all = t27.history_index(episode, step, H)
    rng = np.random.default_rng(SEED)
    n_val = max(1, round(n_ep * VAL_RATIO))
    is_val = np.isin(episode, rng.permutation(n_ep)[:n_val])
    train_rows, val_rows = np.flatnonzero(~is_val), np.flatnonzero(is_val)

    torch.manual_seed(SEED)
    net = t27.build_net(H * frames.shape[1] + cmds.shape[1], actions.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    best, t0 = np.inf, time.time()
    for epoch in range(1, epochs + 1):
        net.train()
        perm = rng.permutation(train_rows)
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            loss = loss_fn(net(torch.from_numpy(t27.make_obs(frames, cmds, idx_all[b]))),
                           torch.from_numpy(actions[b]))
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        va = t27.batch_loss(net, loss_fn, frames, actions, cmds, val_rows, idx_all)
        best = min(best, va)
        if epoch % max(1, epochs // 6) == 0 or epoch == epochs:
            print(f"    [{tag}] 第 {epoch}/{epochs} 轮  验证 MSE {va:.5f}（最好 {best:.5f}）", flush=True)
    return net, best, time.time() - t0


if __name__ == "__main__":
    main()

