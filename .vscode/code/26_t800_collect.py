"""采集老师策略（官方 MNN 模型）的示范数据，供阶段 1 的行为克隆训练使用。

每个推理步（100 Hz）记录：
  72 维单帧观测 + 22 维动作（老师输出）+ 3 维指令 + 回合编号 + 回合内步数
训练时再按 15 帧历史拼成 1083 维观测（见 27_t800_bc_train.py），这样数据量小很多：
一回合 15 s = 1500 步，约 0.6 MB（单帧 72 + 动作 22 + 指令 3，都是 float32）。

为了增加数据多样性（25 里的 run_trial 是确定性的，初始扰动只有 ±0.02 rad，步态几秒后就收敛到同一状态），
每个回合随机化三件事：
  - 速度指令：vx ~ U(VX_RANGE)，默认 0.3~0.6 m/s，避开 ±0.25 m/s 的指令死区
  - 初始关节角扰动：±INIT_NOISE，默认 0.10 rad，比 25 里的 0.02 大 5 倍
  - 行走途中随机推 0~3 次：推力 40~120 N、方向随机、持续 0.2 s，力度控制在不把老师推倒

观测构建、推理、PD 控制、摔倒判断都直接复用 25 的常量和函数，保证和评测脚本完全一致
（常量一律从 25 取，不在本文件里重写一份）。

用法：
  python 26_t800_collect.py --quick                 # 小规模试跑：3 个回合 × 5 s，输出到 data/_smoke/
  python 26_t800_collect.py --episodes 20           # 20 个回合 × 15 s
  python 26_t800_collect.py                         # 默认 200 个回合 × 15 s，输出到 data/bc_teacher.npz

输出：
  data/<name>.npz     frames (N,72) / actions (N,22) / cmds (N,3) / episode (N,) / step (N,)
  data/<name>.csv     每个回合一行：种子、指令 vx、推的次数和力度、是否摔倒、记录步数
"""
import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
DATA_DIR = CODE_DIR / "data"

# ---------- 采集设置 ----------
EPISODES = 200            # 回合数：每回合是一条完整的连续轨迹
DURATION = 15.0           # 每回合仿真时长 (s)
VX_RANGE = (0.3, 0.6)     # 速度指令范围 (m/s)
INIT_NOISE = 0.10         # 初始关节角扰动幅度 (rad)
PUSH_TIME_RANGE = (3.0, 12.0)     # 途中推的时刻范围 (s)
PUSH_FORCE_RANGE = (40.0, 120.0)  # 推力大小 (N)，老师的侧向极限在 150 N 以上
PUSH_DURATION = 0.2       # 每次推持续 (s)
MAX_PUSHES = 3            # 每回合最多推几次
CHUNK_EPISODES = 50       # 每采集这么多回合落盘一次，中途意外退出也不至于全丢

# 25 的文件名以数字开头不能直接 import，用 importlib 加载；加载后当前目录会变成 MNN-CODE
spec = importlib.util.spec_from_file_location("r25", CODE_DIR / "25_t800_robustness.py")
r25 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r25)


def rollout(seed, duration):
    """让老师走一个回合，返回记录和这一回合的元信息。

    循环结构和 25 的 run_trial 保持一致：每 DECIMATION 个仿真步推理一次，
    其余仿真步只做 PD 控制和 mj_step，摔倒判断也在推理步里做。
    """
    model, data = r25.model, r25.data
    rng = np.random.default_rng(seed)
    vx = float(rng.uniform(*VX_RANGE))

    # 随机抽这一回合的推：时刻、方向、大小
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
    frames, actions, cmds = [], [], []
    fell = False

    for step in range(round(duration / r25.DT)):
        if step % r25.DECIMATION == 0:
            R = data.xmat[r25.base_id].reshape(3, 3)
            gravity = R.T @ [0, 0, -1]
            if data.qpos[2] < r25.FALL_HEIGHT or gravity[2] > -np.cos(np.radians(r25.FALL_TILT)):
                fell = True
                break
            # 72 维单帧：关节角偏差 22 + 关节速度 22 + 上一帧动作 22 + 角速度 3 + 重力投影 3
            q, dq = data.qpos[r25.qadr], data.qvel[r25.dadr]
            frame = np.concatenate([(q - r25.DEFAULT_Q)[r25.POLICY], dq[r25.POLICY] * r25.DQ_SCALE, action,
                                    data.qvel[3:6], gravity])
            frame = np.clip(frame, -r25.CLIP, r25.CLIP).astype(np.float32)
            history = [frame] * r25.HISTORY if history is None else history[1:] + [frame]
            yaw_err = np.angle(np.exp(1j * (desired_yaw - r25.yaw_of(data.qpos[3:7]))))
            cmd = np.array([vx * r25.CMD_SCALE, 0.0, np.clip(r25.YAW_KP * yaw_err, -1, 1)], np.float32)
            # 老师推理：这一步的 action 既是记录下来的标签，也是下一帧观测里的"上一帧动作"
            action = np.clip(r25.infer(np.concatenate(history + [cmd])), -r25.CLIP, r25.CLIP)
            q_des = r25.DEFAULT_Q.copy()
            q_des[r25.POLICY] += action * r25.ACTION_SCALE
            frames.append(frame)
            actions.append(action)
            cmds.append(cmd)

        tau = r25.KP * (q_des - data.qpos[r25.qadr]) - r25.KD * data.qvel[r25.dadr]
        data.ctrl[:] = np.clip(tau, -r25.tau_limit, r25.tau_limit)
        fx = fy = 0.0
        for t0, px, py in pushes:
            if t0 <= data.time < t0 + PUSH_DURATION:
                fx, fy = px, py
        data.xfrc_applied[r25.base_id, :2] = (fx, fy)  # 不在推的时候就置 0
        r25.mujoco.mj_step(model, data)

    rec = dict(frames=np.array(frames, np.float32), actions=np.array(actions, np.float32),
               cmds=np.array(cmds, np.float32))
    info = dict(seed=seed, vx=vx, n_push=len(pushes),
                push_force=max([np.hypot(p[1], p[2]) for p in pushes], default=0.0),
                fell=fell, steps=len(frames), t_end=float(data.time))
    return rec, info


def save(out_path, recs, infos):
    """把若干回合拼成一个大数组存成 npz，另外存一份每回合一行的 csv。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.concatenate([r["frames"] for r in recs])
    actions = np.concatenate([r["actions"] for r in recs])
    cmds = np.concatenate([r["cmds"] for r in recs])
    episode = np.concatenate([np.full(len(r["frames"]), i, np.int32) for i, r in enumerate(recs)])
    step = np.concatenate([np.arange(len(r["frames"]), dtype=np.int32) for r in recs])
    np.savez(out_path, frames=frames, actions=actions, cmds=cmds, episode=episode, step=step)

    with open(out_path.with_suffix(".csv"), "w", encoding="utf-8-sig", newline="") as f:
        f.write("回合,种子,指令vx,推的次数,最大推力(N),是否摔倒,记录步数,结束时刻(s)\n")
        for i, info in enumerate(infos):
            f.write(f"{i},{info['seed']},{info['vx']:.3f},{info['n_push']},{info['push_force']:.1f},"
                    f"{'摔倒' if info['fell'] else '没摔'},{info['steps']},{info['t_end']:.3f}\n")
    return frames.shape, out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=EPISODES, help=f"回合数，默认 {EPISODES}")
    parser.add_argument("--duration", type=float, default=DURATION, help=f"每回合时长 (s)，默认 {DURATION}")
    parser.add_argument("--seed0", type=int, default=0, help="第一个回合的种子，第 i 回合用 seed0 + i")
    parser.add_argument("--out", default="bc_teacher.npz", help="输出文件名（放在 data/ 下）")
    parser.add_argument("--quick", action="store_true", help="小规模试跑：3 个回合 × 5 s，输出到 data/_smoke/")
    args = parser.parse_args()
    if args.quick:
        args.episodes, args.duration, args.out = 3, 5.0, "_smoke/smoke.npz"

    out_path = DATA_DIR / args.out
    print(f"采集 {args.episodes} 个回合 × {args.duration} s，指令 vx ∈ {VX_RANGE}，"
          f"初始扰动 ±{INIT_NOISE} rad，途中推 0~{MAX_PUSHES} 次"
          f"（{PUSH_FORCE_RANGE[0]:.0f}~{PUSH_FORCE_RANGE[1]:.0f} N）")
    t0 = time.time()
    recs, infos = [], []
    for i in range(args.episodes):
        rec, info = rollout(args.seed0 + i, args.duration)
        recs.append(rec)
        infos.append(info)
        state = f"摔倒 {info['t_end']:.2f} s" if info["fell"] else "没摔"
        print(f"  回合 {i:>3}  种子 {info['seed']:>3}  vx {info['vx']:.2f}  推 {info['n_push']} 次"
              f"（最大 {info['push_force']:.0f} N）  {state}  记录 {info['steps']} 步  "
              f"累计 {time.time() - t0:.0f} s", flush=True)
        if (i + 1) % CHUNK_EPISODES == 0 and i + 1 < args.episodes:  # 中途落盘，防止意外退出全丢
            save(out_path.with_name(out_path.stem + f"_part{i + 1}.npz"), recs, infos)

    shape, path = save(out_path, recs, infos)
    fell = sum(info["fell"] for info in infos)
    print(f"完成：{len(recs)} 个回合，{shape[0]} 步，摔倒 {fell} 个回合，"
          f"用时 {time.time() - t0:.0f} s，文件 {path}（{path.stat().st_size / 1e6:.1f} MB）")


if __name__ == "__main__":
    main()
