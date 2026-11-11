"""T800 RL 行走策略鲁棒性评测：不开窗口批量仿真，统计 4 类扰动下的摔倒率和速度跟踪误差。

实验（命令行参数选择，默认全跑）：
  speed     速度跟踪：不同 vx / vy 指令下的实际速度、跟踪误差、摔倒率
  push      行走中抗推：0.5 m/s 行走时从前 / 后 / 左 / 右推 0.2 s，找每个方向的最大可恢复推力
  phase     推的时机：侧向推力固定，在一个步态周期内均匀取 50 个时刻推，看结果和步态相位的关系
  dynamics  动力学偏差：分别找摩擦边界、负载边界，再测一个 3 × 3 的摩擦 × 负载组合
  payload   负载细测：40% ~ 45% 每 1% 一档，统计每个关节力矩被限幅的时间占比
  trace     负载 42%（不摔）和 44%（摔）的时间线对比：速度、重心、前倾、踝关节力矩谁先出问题
  latency   延迟和噪声：动作延迟、IMU 观测噪声

用法：
  python 25_t800_robustness.py                 # 全部实验
  python 25_t800_robustness.py push latency    # 只跑指定实验
  python 25_t800_robustness.py --quick speed   # 每组只跑 1 次，先检查流程

结果保存在 .vscode/code/results/：每个实验一个 CSV + 一张 PNG。
控制部分（obs、推理、PD）和 20_t800_mnn_walk_sdk.py 一致。
"""
import argparse
import csv
import os
import time
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 不弹窗口，直接存图
import matplotlib.pyplot as plt
import MNN
import mujoco
import numpy as np

CODE_DIR = Path(__file__).resolve().parent
ROOT_DIR = CODE_DIR.parents[1]
RESULT_DIR = CODE_DIR / "results"

# ---------- 评测设置 ----------
DURATION = 15.0        # 每次仿真时长 (s)
WARMUP = 3.0           # 前 3 s 是起步阶段，不计入速度统计
REPEATS = 5            # 每组条件重复次数：每次初始关节角随机扰动，推的时机也随机
INIT_NOISE = 0.02      # 初始关节角随机扰动幅度 (rad)
FALL_HEIGHT = 0.6      # 根部高度低于它算摔倒 (m)，站立时约 1.0 m
FALL_TILT = 45.0       # 机身倾斜超过它算摔倒 (deg)
PASS_RATE = 0.8        # 成功率 >= 80% 算"能恢复"

SPEED_CMDS = ([(round(v, 1) + 0.0, 0.0) for v in np.arange(-0.5, 1.01, 0.1)]                    # vx：-0.5 ~ 1.0，每 0.1 一档（+0.0 去掉 -0.0）
              + [(0.0, round(v, 1)) for v in np.arange(-0.4, 0.41, 0.1) if abs(v) > 1e-9])    # vy：-0.4 ~ 0.4，每 0.1 一档
PUSH_DIRS = {"向前 (+x)": (1, 0), "向后 (-x)": (-1, 0), "向左 (+y)": (0, 1), "向右 (-y)": (0, -1)}
PUSH_FORCES = np.arange(50, 801, 25)   # 推力 (N)，连续 2 档全部摔倒就提前停止
PUSH_TIME, PUSH_DURATION = 5.0, 0.2    # 在 5 ~ 6 s 之间随机时刻推 0.2 s，覆盖不同的步态相位
PUSH_REPEATS = 10      # 抗推实验每档推力的重复次数（比其他实验多，因为结果和推的时机关系很大）
PHASE_FORCES = {"向左 (+y)": 400, "向右 (-y)": 350}  # 推的时机实验：取侧向过渡区中间的推力（向右 250 N 太轻，改为 350 N）
PHASE_POINTS = 50      # 在一个步态周期内均匀取 50 个推的时刻
CONTACT_RATIO = 0.2   # 一只脚承担的力超过整机重量的 20% 才算这只脚在支撑（只是脚尖蹭地不算）
DYN_REPEATS = 10       # 动力学实验每组的重复次数（第一版 5 次时临界点附近的结果不稳定）
FRICTION_SCALES = [0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.35, 0.4, 0.45, 0.5, 1.0]  # 摩擦边界：不加负载
PAYLOADS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6]  # 负载边界：正常摩擦，占整机质量的比例
COMBO_FRICTIONS = [0.3, 0.5, 1.0]      # 组合测试：摩擦 × 负载
COMBO_PAYLOADS = [0.0, 0.2, 0.4]
PAYLOAD_FINE = [0.0, 0.2, 0.3, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45]  # 负载细测：40% ~ 45% 每 1% 一档，前面几档作对照
DELAYS_MS = [0, 10, 20, 30, 40, 45, 50, 55, 60]
NOISE_LEVELS = [(0.0, 0.0), (0.1, 0.05), (0.2, 0.1), (0.25, 0.125), (0.3, 0.15), (0.35, 0.175), (0.4, 0.2)]  # IMU 噪声标准差：(角速度 rad/s, 重力投影)

# ---------- SDK 部署参数（同 20） ----------
DEFAULT_Q = np.array([-0.06, 0, 0, 0.12, -0.06, 0,  -0.06, 0, 0, 0.12, -0.06, 0,  0,
                      0, 0.15, 0, -0.25, 0,  0, -0.15, 0, -0.25, 0,  0, 0])
KP = np.array([180, 100, 100, 180, 40, 40,  180, 100, 100, 180, 40, 40,  100,
               60, 50, 50, 60, 50,  60, 50, 50, 60, 50,  100, 100], float)
KD = np.array([5, 3, 3, 5, 0.3, 0.3,  5, 3, 3, 5, 0.3, 0.3,  5,
               0.3, 0.3, 0.3, 0.3, 0.3,  0.3, 0.3, 0.3, 0.3, 0.3,  1, 1], float)
SOFT_TORQUE = np.array([0.8] * 12 + [0.9] * 13)
POLICY = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
ACTION_SCALE = np.array([0.5, 0.2, 0.2, 0.5, 0.5, 0.2,  0.5, 0.2, 0.2, 0.5, 0.5, 0.2,
                         0.2, 0.2, 0.05, 0.2, 0.05,  0.2, 0.2, 0.05, 0.2, 0.05])
DQ_SCALE, CMD_SCALE, YAW_KP = 0.05, 2.0, 2.0
HISTORY, CLIP, DECIMATION = 15, 100.0, 5

os.chdir(ROOT_DIR)  # MuJoCo 和 MNN 都打不开含中文的绝对路径
model = mujoco.MjModel.from_xml_path("t800.xml")
data = mujoco.MjData(model)
os.chdir(CODE_DIR / "MNN-CODE")
net = MNN.Interpreter("policy/t800_260318_150533_60000.mnn")
session = net.createSession()


def infer(obs):
    x = MNN.Tensor((1, obs.size), MNN.Halide_Type_Float, obs.reshape(1, -1), MNN.Tensor_DimensionType_Caffe)
    net.getSessionInput(session).copyFrom(x)
    net.runSession(session)
    y = MNN.Tensor((1, 22), MNN.Halide_Type_Float, np.zeros((1, 22), np.float32), MNN.Tensor_DimensionType_Caffe)
    net.getSessionOutput(session).copyToHostTensor(y)
    return np.array(y.getData(), np.float32)


def yaw_of(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


qadr = np.array([model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in range(model.nu)])
dadr = np.array([model.jnt_dofadr[model.actuator_trnid[i, 0]] for i in range(model.nu)])
tau_limit = model.actuator_ctrlrange[:, 1] * SOFT_TORQUE
base_id = model.body("LINK_BASE").id
DT = model.opt.timestep
FRICTION0 = model.geom_friction[:, 0].copy()
MASS0, INERTIA0 = model.body_mass.copy(), model.body_inertia.copy()
ROBOT_MASS = model.body_subtreemass[base_id]
FOOT_L = {model.body("LINK_ANKLE_ROLL_L").id, model.body("LINK_FOOT_L").id}
FOOT_R = {model.body("LINK_ANKLE_ROLL_R").id, model.body("LINK_FOOT_R").id}
FOOT_BODY = [model.body("LINK_FOOT_L").id, model.body("LINK_FOOT_R").id]
ANKLE_P = [i for i in range(model.nu) if "ANKLE_PITCH" in model.actuator(i).name]  # 左、右踝 pitch 执行器
n_trials = 0  # 累计仿真次数


def foot_forces():
    """左、右脚和地面之间的法向力之和 (N)。地面是挂在 world (body 0) 上的 geom。"""
    left, right, f6 = 0.0, 0.0, np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        bodies = {model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]}
        if 0 not in bodies:
            continue
        mujoco.mj_contactForce(model, data, i, f6)
        if bodies & FOOT_L:
            left += abs(f6[0])
        elif bodies & FOOT_R:
            right += abs(f6[0])
    return left, right


def foot_stance():
    """当前哪只脚着地：'左脚' / '右脚' / '双脚' / '腾空'。"""
    threshold = CONTACT_RATIO * ROBOT_MASS * 9.81
    left, right = (f > threshold for f in foot_forces())
    return "双脚" if left and right else "左脚" if left else "右脚" if right else "腾空"


def set_dynamics(friction_scale=1.0, payload=0.0):
    """修改摩擦和负载，都基于原始值计算，可以反复调用；不传参数就是还原。"""
    # MuJoCo 中两个 geom 接触时取两者摩擦系数的较大值，所以地面和机器人的摩擦要一起缩放
    model.geom_friction[:, 0] = FRICTION0 * friction_scale
    # 负载加在机身上，惯量按同样比例放大（相当于机身变重了）
    model.body_mass[:] = MASS0
    model.body_inertia[:] = INERTIA0
    model.body_mass[base_id] += payload * ROBOT_MASS
    model.body_inertia[base_id] *= model.body_mass[base_id] / MASS0[base_id]
    mujoco.mj_setConst(model, data)  # 重新计算子树质量等派生量


def run_trial(vx=0.5, vy=0.0, seed=0, push=None, delay_ms=0, noise=(0.0, 0.0), t_push=None, foot_log=None, sat=None,
              trace=None, policy=None):
    """不开窗口跑一次仿真。push = (fx, fy) 是世界坐标系下的推力 (N)。
    t_push 指定推的时刻 (s)，不指定就在 5 ~ 6 s 之间随机；foot_log 传入列表时，每个推理周期记录 (时刻, 左脚力, 右脚力)；
    sat 传入长度为 nu 的数组时，累加每个关节 PD 力矩达到上限（被限幅）的仿真步数；
    trace 传入列表时，每个仿真步记录 (时刻, 机身 vx, 前倾角 deg, 重心超前双脚中点的距离 m, 根部高度,
    左踝 pitch 需求力矩 / 上限, 右踝 pitch 需求力矩 / 上限)，比值 >= 1 表示被限幅；
    policy 传入一个函数 obs(1083 维数组) -> action(22 维数组) 时，用它代替官方策略做推理，
    默认 None 表示用本地 MNN 模型（老师），此时行为和以前完全一样。
    返回 (是否摔倒, 结束时刻, 实际平均速度 vx, vy, 推的时刻, 推的瞬间哪只脚着地, 平均前倾角 deg)；
    摔倒时速度和前倾角为 nan；不推时推的时刻和着地脚为 nan 和 None。"""
    global n_trials
    n_trials += 1
    rng = np.random.default_rng(seed)
    mujoco.mj_resetData(model, data)
    data.qpos[qadr] = DEFAULT_Q
    data.qpos[qadr[POLICY]] += rng.uniform(-INIT_NOISE, INIT_NOISE, len(POLICY))
    mujoco.mj_forward(model, data)

    history, action, q_des = None, np.zeros(22, np.float32), DEFAULT_Q.copy()
    desired_yaw = yaw_of(data.qpos[3:7])
    delay = round(delay_ms / 1000 / DT)
    q_buf = deque([DEFAULT_Q.copy()] * (delay + 1), maxlen=delay + 1)  # 动作延迟：执行 delay 步之前的目标角
    t_random = PUSH_TIME + rng.uniform(0, 1.0)  # 即使指定了 t_push 也照样抽一次，保证后面的随机数序列不变
    t_push = t_random if t_push is None else t_push
    sigma_w, sigma_g = noise
    xy_start, stance, pitches = None, None, []

    for step in range(round(DURATION / DT)):
        if step % DECIMATION == 0:
            R = data.xmat[base_id].reshape(3, 3)
            gravity = R.T @ [0, 0, -1]
            if data.qpos[2] < FALL_HEIGHT or gravity[2] > -np.cos(np.radians(FALL_TILT)):
                return True, data.time, np.nan, np.nan, t_push if push else np.nan, stance, np.nan
            if data.time >= WARMUP:
                w, x, y, z = data.qpos[3:7]
                pitches.append(np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))))  # 正值 = 机身前倾
            if foot_log is not None:
                foot_log.append((data.time, *foot_forces()))
            # IMU 噪声只加在策略看到的观测上，不影响摔倒判断
            ang_vel = data.qvel[3:6] + rng.normal(0, sigma_w, 3)
            grav_obs = gravity + rng.normal(0, sigma_g, 3)
            q, dq = data.qpos[qadr], data.qvel[dadr]
            frame = np.concatenate([(q - DEFAULT_Q)[POLICY], dq[POLICY] * DQ_SCALE, action, ang_vel, grav_obs])
            frame = np.clip(frame, -CLIP, CLIP).astype(np.float32)
            history = [frame] * HISTORY if history is None else history[1:] + [frame]
            yaw_err = np.angle(np.exp(1j * (desired_yaw - yaw_of(data.qpos[3:7]))))
            cmd = np.array([vx * CMD_SCALE, vy * CMD_SCALE, np.clip(YAW_KP * yaw_err, -1, 1)], np.float32)
            obs = np.concatenate(history + [cmd])
            # 默认用老师（MNN）；传了 policy 就用它代替，接口是 obs(1083 维) -> action(22 维)
            act = infer(obs) if policy is None else policy(obs)
            action = np.clip(np.asarray(act, np.float32), -CLIP, CLIP)
            q_des = DEFAULT_Q.copy()
            q_des[POLICY] += action * ACTION_SCALE

        q_buf.append(q_des)
        tau = KP * (q_buf[0] - data.qpos[qadr]) - KD * data.qvel[dadr]
        data.ctrl[:] = np.clip(tau, -tau_limit, tau_limit)
        if sat is not None:
            sat += np.abs(tau) >= tau_limit
        if trace is not None:
            w, x, y, z = data.qpos[3:7]
            feet_x = data.xpos[FOOT_BODY, 0].mean()
            trace.append((data.time, data.qvel[0], np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))),
                          data.subtree_com[base_id, 0] - feet_x, data.qpos[2],
                          *(np.abs(tau[ANKLE_P]) / tau_limit[ANKLE_P])))
        if push is not None:
            pushing = t_push <= data.time < t_push + PUSH_DURATION
            if pushing and stance is None:
                stance = foot_stance()  # 记下开始推的那一刻哪只脚着地
            data.xfrc_applied[base_id, :2] = push if pushing else (0.0, 0.0)
        mujoco.mj_step(model, data)
        if xy_start is None and data.time >= WARMUP:
            xy_start = data.qpos[:2].copy()

    # 航向闭环让机器人一直朝初始方向（+x），所以世界坐标系的 x / y 速度就是前进 / 侧向速度
    v = (data.qpos[:2] - xy_start) / (data.time - WARMUP)
    return False, data.time, v[0], v[1], t_push if push else np.nan, stance, np.mean(pitches)


# ---------- 结果保存和画图 ----------
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]  # 中文字体
plt.rcParams["axes.unicode_minus"] = False
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # 分类色，按固定顺序使用


def save_csv(name, header, rows):
    RESULT_DIR.mkdir(exist_ok=True)
    with open(RESULT_DIR / f"{name}.csv", "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig 方便 Excel 打开
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(RESULT_DIR / f"{name}.png", dpi=150)
    plt.close(fig)
    print(f"  -> results/{name}.csv, results/{name}.png")


def style(ax):
    ax.grid(alpha=0.3, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)


def fall_rate(results):
    return np.mean([r[0] for r in results])


def mean_speed(results, i):
    """没摔倒的那几次的平均速度；全摔了返回 nan。"""
    v = [r[i] for r in results if not r[0]]
    return np.mean(v) if v else np.nan


# ---------- 实验 ① 速度跟踪 ----------
def exp_speed():
    rows = []
    for vx, vy in SPEED_CMDS:
        res = [run_trial(vx, vy, seed=s) for s in range(REPEATS)]
        avx, avy = mean_speed(res, 2), mean_speed(res, 3)
        err = np.hypot(avx - vx, avy - vy)
        rows.append([vx, vy, round(avx, 3), round(avy, 3), round(err, 3), fall_rate(res)])
        print(f"  指令 ({vx:+.1f}, {vy:+.1f})  实际 ({avx:+.3f}, {avy:+.3f})  误差 {err:.3f} m/s  摔倒率 {fall_rate(res):.0%}")
    save_csv("speed", ["指令vx", "指令vy", "实际vx", "实际vy", "跟踪误差(m/s)", "摔倒率"], rows)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, k, name in [(axes[0], 0, "vx"), (axes[1], 1, "vy")]:
        sel = [r for r in rows if r[1 - k] == 0]  # 前后方向只看 vy = 0 的组，侧向只看 vx = 0 的组
        cmd, act = [r[k] for r in sel], [r[2 + k] for r in sel]
        lim = [min(cmd) - 0.2, max(cmd) + 0.2]
        ax.plot(lim, lim, "--", color="#9a9893", linewidth=1.5, label="理想跟踪")
        ax.plot(cmd, act, "o", color=COLORS[0], markersize=8, label="实际速度")
        ax.set(xlabel=f"指令 {name} (m/s)", ylabel=f"实际 {name} (m/s)", title=f"{name} 速度跟踪")
        ax.legend(frameon=False)
        style(ax)
    save_fig(fig, "speed")


# ---------- 实验 ② 行走中抗推 ----------
def exp_push():
    rows, trials, curves, limits = [], [], {}, {}
    for (name, (ux, uy)), color in zip(PUSH_DIRS.items(), COLORS):
        rates, zero_count = [], 0
        for f in PUSH_FORCES:
            res = [run_trial(0.5, 0.0, seed=s, push=(f * ux, f * uy)) for s in range(PUSH_REPEATS)]
            ok = round(1 - fall_rate(res), 3)
            rates.append(ok)
            rows.append([name, f, ok])
            # 每一次推的明细：推的时刻、当时哪只脚着地、有没有摔
            trials += [[name, f, s, round(r[4], 3), r[5], "摔倒" if r[0] else "恢复"] for s, r in enumerate(res)]
            zero_count = zero_count + 1 if ok == 0 else 0
            if zero_count == 2:
                break
        forces = PUSH_FORCES[:len(rates)]
        # 最大可恢复推力：从小到大第一次成功率低于 80% 之前的那一档
        passed = np.flatnonzero(np.array(rates) < PASS_RATE)
        first_fail = passed[0] if passed.size else len(rates)
        limits[name] = forces[first_fail - 1] if first_fail > 0 else 0
        curves[name] = (forces, rates, color)
        print(f"  {name}  最大可恢复推力 {limits[name]} N（推 {PUSH_DURATION} s，成功率 >= {PASS_RATE:.0%}）")
    save_csv("push", ["方向", "推力(N)", "成功率"], rows)
    save_csv("push_limit", ["方向", "最大可恢复推力(N)"], list(limits.items()))
    save_csv("push_trials", ["方向", "推力(N)", "种子", "推的时刻(s)", "着地脚", "结果"], trials)
    plot_push_stance(trials)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, (forces, rates, color) in curves.items():
        ax.plot(forces, np.array(rates) * 100, "-o", color=color, linewidth=2, markersize=6,
                label=f"{name}：{limits[name]} N")
    ax.axhline(PASS_RATE * 100, color="#9a9893", linestyle="--", linewidth=1)
    ax.set(xlabel=f"推力 (N)，持续 {PUSH_DURATION} s", ylabel="恢复成功率 (%)", ylim=(-5, 105),
           title="0.5 m/s 行走中抗推测试")
    ax.legend(frameon=False, title="方向：最大可恢复推力")
    style(ax)
    save_fig(fig, "push")


def plot_push_stance(trials):
    """按"推的瞬间哪只脚着地"分组，比较过渡区内的恢复成功率。
    过渡区：从第一次出现摔倒的推力，到最后一次还有恢复的推力。这段推力下结果最不确定，最能看出着地脚的影响。"""
    stances = ["左脚", "右脚", "双脚"]
    table, rows = {}, []
    for name in PUSH_DIRS:
        tr = [t for t in trials if t[0] == name]
        fail_f = [t[1] for t in tr if t[5] == "摔倒"]
        ok_f = [t[1] for t in tr if t[5] == "恢复"]
        if not fail_f or not ok_f:
            continue
        lo, hi = min(fail_f), max(ok_f)
        if lo > hi:  # 摔倒和恢复的推力没有重叠，没有过渡区
            continue
        zone = [t for t in tr if lo <= t[1] <= hi]
        table[name] = {}
        for s in stances:
            g = [t for t in zone if t[4] == s]
            rate = np.mean([t[5] == "恢复" for t in g]) if g else np.nan
            table[name][s] = (rate, len(g))
            rows.append([name, f"{lo}~{hi}", s, len(g), round(rate, 3) if g else ""])
        others = len(zone) - sum(n for _, n in table[name].values())
        print(f"  {name}  过渡区 {lo}~{hi} N：" + "，".join(
            f"{s} {r:.0%} ({n} 次)" for s, (r, n) in table[name].items() if n) + (f"，其他 {others} 次" if others else ""))
    save_csv("push_stance", ["方向", "过渡区推力(N)", "着地脚", "次数", "恢复成功率"], rows)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    width = 0.26
    for k, (s, color) in enumerate(zip(stances, COLORS)):
        for i, name in enumerate(table):
            rate, n = table[name][s]
            if n == 0:
                continue
            x = i + (k - 1) * width
            ax.bar(x, rate * 100, width * 0.9, color=color, label=s if i == 0 else None)
            ax.text(x, rate * 100 + 2, f"n={n}", ha="center", fontsize=8, color="#52514e")
    ax.set_xticks(range(len(table)), [f"{n}\n{min(t[1] for t in trials if t[0] == n and t[5] == '摔倒')}~"
                                      f"{max(t[1] for t in trials if t[0] == n and t[5] == '恢复')} N" for n in table])
    ax.set(ylabel="恢复成功率 (%)", ylim=(0, 110), title="过渡区内：推的瞬间哪只脚着地 vs 恢复成功率")
    ax.legend(frameon=False, title="推的瞬间着地脚", ncol=3, loc="upper right")
    style(ax)
    save_fig(fig, "push_stance")


# ---------- 实验 ②-b 推的时机：在一个步态周期内均匀取时刻推 ----------
def exp_push_phase():
    """推力固定在侧向过渡区内，只改变推的时刻。初始状态（种子 0）固定，所以推之前的步态每次都一模一样，
    推的时刻就直接对应步态相位。"""
    # 1) 先不推跑一次，记录左右脚受力，找出步态周期
    ref = []
    run_trial(0.5, 0.0, seed=0, foot_log=ref)
    t, fl, fr = np.array(ref).T
    threshold = CONTACT_RATIO * ROBOT_MASS * 9.81
    landing = t[1:][(fl[1:] > threshold) & (fl[:-1] <= threshold)]  # 左脚落地时刻
    period = np.median(np.diff(landing[landing > WARMUP]))
    t0 = landing[landing >= PUSH_TIME][0]                             # 从 5 s 之后左脚第一次落地开始取一个周期
    times = t0 + np.arange(PHASE_POINTS) / PHASE_POINTS * period
    print(f"  步态周期 {period:.3f} s，取 {t0:.3f} ~ {t0 + period:.3f} s 之间的 {PHASE_POINTS} 个时刻")

    # 2) 在每个时刻推一次
    rows, results = [], {}
    for name, f in PHASE_FORCES.items():
        ux, uy = PUSH_DIRS[name]
        results[name] = []
        for tp in times:
            r = run_trial(0.5, 0.0, seed=0, push=(f * ux, f * uy), t_push=tp)
            phase = (tp - t0) / period * 100
            results[name].append((phase, not r[0]))
            rows.append([name, f, round(tp, 3), round(phase, 1), r[5], "摔倒" if r[0] else "恢复"])
        by_stance = {}
        for row in rows:
            if row[0] == name:
                by_stance.setdefault(row[4], []).append(row[5] == "恢复")
        print(f"  {name} {f} N：总成功率 {np.mean([ok for _, ok in results[name]]):.0%}；" +
              "，".join(f"{s} {np.mean(v):.0%} ({len(v)} 个时刻)" for s, v in by_stance.items()))
    save_csv("push_phase", ["方向", "推力(N)", "推的时刻(s)", "步态相位(%)", "着地脚", "结果"], rows)

    # 3) 画图：上面是一个周期内左右脚受力，下面是在每个相位推的结果
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True, height_ratios=[3, 2])
    sel = (t >= t0) & (t <= t0 + period)
    ph = (t[sel] - t0) / period * 100
    ax1.plot(ph, fl[sel], color=COLORS[0], linewidth=2, label="左脚")
    ax1.plot(ph, fr[sel], color=COLORS[1], linewidth=2, label="右脚")
    ax1.axhline(threshold, color="#9a9893", linestyle="--", linewidth=1)
    ax1.text(100, threshold, " 支撑阈值", va="center", fontsize=8, color="#52514e")
    ax1.set(ylabel="脚底法向力 (N)", title=f"推的时机 vs 结果（一个步态周期 {period:.2f} s，推 {PUSH_DURATION} s）")
    ax1.legend(frameon=False, ncol=2, loc="upper right")
    style(ax1)
    for k, (name, res) in enumerate(results.items()):
        ph_ok = [p for p, ok in res if ok]
        ph_fail = [p for p, ok in res if not ok]
        ax2.scatter(ph_ok, [k] * len(ph_ok), s=50, color="#1baf7a", edgecolor="white", label="恢复" if k == 0 else None)
        ax2.scatter(ph_fail, [k] * len(ph_fail), s=50, color="#e34948", marker="x", linewidths=2,
                    label="摔倒" if k == 0 else None)
    ax2.set_yticks(range(len(results)), [f"{n} {PHASE_FORCES[n]} N" for n in results])
    ax2.set(xlabel="开始推的时刻在步态周期中的位置 (%，0 = 左脚落地)", ylim=(-0.6, len(results) - 0.4))
    ax2.legend(frameon=False, ncol=2, loc="upper right", bbox_to_anchor=(1, 1.25))
    style(ax2)
    save_fig(fig, "push_phase")


# ---------- 实验 ③ 动力学偏差：摩擦 × 负载 ----------
DYN_HEADER = ["摩擦缩放", "负载比例", "负载(kg)", "摔倒率", "实际vx(指令0.5)", "平均前倾角(deg)"]


def dyn_condition(fs, payload):
    """在一组摩擦 / 负载条件下跑 DYN_REPEATS 次，返回 CSV 的一行。速度和前倾角只统计没摔的那几次。"""
    set_dynamics(fs, payload)
    res = [run_trial(0.5, 0.0, seed=s) for s in range(DYN_REPEATS)]
    fr, avx, pitch = round(fall_rate(res), 3), mean_speed(res, 2), mean_speed(res, 6)
    print(f"  摩擦 ×{fs:<5} 负载 +{payload:.0%} ({payload * ROBOT_MASS:4.1f} kg)  "
          f"摔倒率 {fr:>4.0%}  实际 vx {avx:.3f}  前倾 {pitch:+.2f}°")
    return [fs, payload, round(payload * ROBOT_MASS, 1), fr, round(avx, 3), round(pitch, 2)]


def exp_dynamics():
    print("  --- A. 摩擦边界（不加负载）---")
    fric_rows = [dyn_condition(fs, 0.0) for fs in FRICTION_SCALES]
    print("  --- B. 负载边界（正常摩擦）---")
    load_rows = [dyn_condition(1.0, p) for p in PAYLOADS]
    print("  --- C. 组合（摩擦 × 负载）---")
    combo_rows = [dyn_condition(fs, p) for p in COMBO_PAYLOADS for fs in COMBO_FRICTIONS]
    grid = np.array([r[3] for r in combo_rows]).reshape(len(COMBO_PAYLOADS), len(COMBO_FRICTIONS))
    set_dynamics()  # 还原
    save_csv("dynamics_friction", DYN_HEADER, fric_rows)
    save_csv("dynamics_payload", DYN_HEADER, load_rows)
    save_csv("dynamics", DYN_HEADER, combo_rows)
    plot_dynamics_curves(fric_rows, load_rows)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    im = ax.imshow(grid * 100, cmap="Blues", vmin=0, vmax=100, aspect="auto", origin="lower")
    for i in range(len(COMBO_PAYLOADS)):
        for j in range(len(COMBO_FRICTIONS)):
            ax.text(j, i, f"{grid[i, j]:.0%}", ha="center", va="center",
                    color="white" if grid[i, j] > 0.5 else "#0b0b0b")
    ax.set_xticks(range(len(COMBO_FRICTIONS)), [f"×{f}" for f in COMBO_FRICTIONS])
    ax.set_yticks(range(len(COMBO_PAYLOADS)), [f"+{p:.0%}\n({p * ROBOT_MASS:.0f} kg)" for p in COMBO_PAYLOADS])
    ax.set(xlabel="摩擦系数缩放", ylabel="机身负载", title="0.5 m/s 行走摔倒率：摩擦 × 负载")
    fig.colorbar(im, ax=ax, label="摔倒率 (%)")
    save_fig(fig, "dynamics")


ACT_NAMES = [model.actuator(i).name.replace("motor_", "").replace("J_", "") for i in range(model.nu)]


def exp_payload_fine():
    """负载 40% ~ 45% 每 1% 一档，统计各关节力矩被限幅的时间占比，验证"负载太重 -> 某个关节力矩不够"。
    限幅占比 = 力矩达到上限的仿真步数 / 总仿真步数（摔倒的那次只算到摔倒为止）。"""
    rows, sat_table = [], []
    for p in PAYLOAD_FINE:
        set_dynamics(1.0, p)
        sat, steps, res = np.zeros(model.nu), 0, []
        for s in range(DYN_REPEATS):
            r = run_trial(0.5, 0.0, seed=s, sat=sat)
            res.append(r)
            steps += round(r[1] / DT)
        ratio = sat / steps * 100
        sat_table.append(ratio)
        top = np.argsort(ratio)[::-1][:3]
        fr, avx = round(fall_rate(res), 3), mean_speed(res, 2)
        t_fall = np.mean([r[1] for r in res if r[0]]) if fr > 0 else np.nan
        rows.append([p, round(p * ROBOT_MASS, 1), fr, round(avx, 3), round(t_fall, 2)]
                    + [round(v, 2) for v in ratio])
        print(f"  负载 +{p:.0%} ({p * ROBOT_MASS:4.1f} kg)  摔倒率 {fr:>4.0%}  实际 vx {avx:.3f}  "
              f"平均摔倒时刻 {t_fall:5.2f} s  限幅最多：" + "，".join(f"{ACT_NAMES[i]} {ratio[i]:.1f}%" for i in top))
    set_dynamics()
    save_csv("payload_fine", ["负载比例", "负载(kg)", "摔倒率", "实际vx(指令0.5)", "平均摔倒时刻(s)"]
             + [f"限幅占比%_{n}" for n in ACT_NAMES], rows)

    # 只画限幅最多的 4 个关节（按最重负载那一档排序），其余关节几乎为 0
    sat_table = np.array(sat_table)
    top = np.argsort(sat_table[-1])[::-1][:4]
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    x = [p * 100 for p in PAYLOAD_FINE]
    axes[0].plot(x, [r[2] * 100 for r in rows], "-o", color=COLORS[1], linewidth=2, markersize=7)
    axes[0].set(ylabel="摔倒率 (%)", ylim=(-5, 105), title=f"负载细测：正常摩擦，指令 0.5 m/s，每档 {DYN_REPEATS} 次")
    for k, i in enumerate(top):
        axes[1].plot(x, sat_table[:, i], "-o", color=COLORS[k], linewidth=2, markersize=6, label=ACT_NAMES[i])
    axes[1].set(xlabel=f"机身负载（% 整机质量，100% = {ROBOT_MASS:.0f} kg）", ylabel="力矩被限幅的时间占比 (%)")
    axes[1].legend(frameon=False, title="限幅最多的关节")
    for ax in axes:
        style(ax)
    save_fig(fig, "payload_fine")


TRACE_PAYLOADS = [0.42, 0.44]  # 时间线对比：刚好不摔 vs 一定摔
TRACE_SEED = 0


def first_time(t, cond):
    """cond 第一次为 True 的时刻，没有就返回 nan。"""
    idx = np.flatnonzero(cond)
    return t[idx[0]] if idx.size else np.nan


def exp_payload_trace():
    """记录摔倒前各个量随时间的变化，判断踝关节限幅是摔倒的原因还是结果。
    思路：如果踝关节先长时间限幅、之后机身才开始失控，说明力矩不够是原因；
    如果机身速度 / 重心先跑偏、踝关节最后才被迫打满，说明限幅只是结果。"""
    traces = {}
    for p in TRACE_PAYLOADS:
        set_dynamics(1.0, p)
        tr = []
        r = run_trial(0.5, 0.0, seed=TRACE_SEED, trace=tr)
        traces[p] = (np.array(tr), r)
    set_dynamics()

    rows = []
    ref = traces[TRACE_PAYLOADS[0]][0]
    ref_vx, ref_com = ref[ref[:, 0] >= WARMUP, 1].mean(), ref[ref[:, 0] >= WARMUP, 3]
    com_band = np.abs(ref_com - ref_com.mean()).max()  # 不摔的那组里，重心偏移的正常波动范围
    for p, (tr, r) in traces.items():
        t, vx, pitch, com, z = tr[:, 0], tr[:, 1], tr[:, 2], tr[:, 3], tr[:, 4]
        ankle = tr[:, 5:7].max(axis=1)
        # 速度用 0.5 s 滑动平均（单步速度在步态里起伏很大）
        win = round(0.5 / DT)
        vx_avg = np.convolve(vx, np.ones(win) / win, mode="same")
        events = {
            "速度持续超出不摔组 20%": first_time(t, (t > WARMUP) & (vx_avg > ref_vx * 1.2)),
            "重心超出不摔组的正常范围": first_time(t, (t > WARMUP) & (np.abs(com - ref_com.mean()) > com_band * 1.5)),
            "踝关节第一次限幅": first_time(t, ankle >= 1.0),
            "前倾超过 10°": first_time(t, pitch > 10),
            "判定摔倒": r[1] if r[0] else np.nan,
        }
        print(f"  负载 +{p:.0%}（{p * ROBOT_MASS:.1f} kg）：{'摔倒' if r[0] else '没摔'}")
        for name, tm in sorted(events.items(), key=lambda kv: (np.isnan(kv[1]), kv[1])):
            print(f"    {name:<14} {'—' if np.isnan(tm) else f'{tm:6.2f} s'}")
            rows.append([p, name, "" if np.isnan(tm) else round(tm, 3)])
        traces[p] = (tr, r, vx_avg, events)
    save_csv("payload_trace", ["负载比例", "事件", "时刻(s)"], rows)
    plot_payload_trace(traces, ref_vx)


def plot_payload_trace(traces, ref_vx):
    fig, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
    for (p, (tr, r, vx_avg, events)), color in zip(traces.items(), [COLORS[0], COLORS[1]]):
        t = tr[:, 0]
        label = f"+{p:.0%} ({p * ROBOT_MASS:.1f} kg)，{'摔倒' if r[0] else '没摔'}"
        axes[0].plot(t, vx_avg, color=color, linewidth=2, label=label)
        axes[1].plot(t, tr[:, 3] * 100, color=color, linewidth=1.5)
        axes[2].plot(t, tr[:, 2], color=color, linewidth=1.5)
        axes[3].plot(t, tr[:, 5:7].max(axis=1) * 100, color=color, linewidth=1, alpha=0.8)
        if r[0]:
            for ax in axes:
                ax.axvline(r[1], color=color, linestyle=":", linewidth=1.5)
    axes[0].axhline(0.5, color="#9a9893", linestyle="--", linewidth=1)
    axes[0].set(ylabel="机身 vx (m/s)\n0.5 s 滑动平均", title="负载临界点附近：摔倒前各个量的时间顺序（点线 = 判定摔倒）")
    axes[0].legend(frameon=False, loc="upper left")
    axes[1].set(ylabel="重心超前双脚\n中点 (cm)")
    axes[2].set(ylabel="前倾角 (deg)")
    axes[3].axhline(100, color="#9a9893", linestyle="--", linewidth=1)
    axes[3].set(ylabel="踝 pitch 需求力矩\n/ 上限 (%)", xlabel="仿真时间 (s)")
    for ax in axes:
        style(ax)
    save_fig(fig, "payload_trace")


def plot_dynamics_curves(fric_rows, load_rows):
    """摩擦边界、负载边界各一张图。摔倒率、速度、前倾角单位不同，分成上下子图，不共用纵轴。"""
    def panel(ax, x, y, color, ylabel, ref=None, ylim=None):
        ax.plot(x, y, "-o", color=color, linewidth=2, markersize=7)
        if ref is not None:
            ax.axhline(ref, color="#9a9893", linestyle="--", linewidth=1)
        ax.set(ylabel=ylabel, ylim=ylim)
        style(ax)

    # 摩擦：横轴按档位等距排开（0.15~0.3 很密，1.0 离得很远）
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    x = np.arange(len(fric_rows))
    panel(axes[0], x, [r[3] * 100 for r in fric_rows], COLORS[1], "摔倒率 (%)", ylim=(-5, 105))
    panel(axes[1], x, [r[4] for r in fric_rows], COLORS[0], "实际 vx (m/s)\n只统计没摔的", ref=0.5)
    axes[0].set_title(f"摩擦边界：不加负载，指令 0.5 m/s，每档 {DYN_REPEATS} 次")
    axes[1].set_xticks(x, [f"×{r[0]}" for r in fric_rows])
    axes[1].set_xlabel("摩擦系数缩放（横轴按档位排列，不等距）")
    save_fig(fig, "dynamics_friction")

    # 负载：多一张前倾角，用来检验"负载越重越前倾、走得越快"
    fig, axes = plt.subplots(3, 1, figsize=(8, 7.5), sharex=True)
    x = [r[1] * 100 for r in load_rows]
    panel(axes[0], x, [r[3] * 100 for r in load_rows], COLORS[1], "摔倒率 (%)", ylim=(-5, 105))
    panel(axes[1], x, [r[4] for r in load_rows], COLORS[0], "实际 vx (m/s)\n只统计没摔的", ref=0.5)
    panel(axes[2], x, [r[5] for r in load_rows], COLORS[2], "平均前倾角 (deg)\n正 = 前倾")
    axes[0].set_title(f"负载边界：正常摩擦，指令 0.5 m/s，每档 {DYN_REPEATS} 次")
    axes[2].set_xlabel(f"机身负载（% 整机质量，100% = {ROBOT_MASS:.0f} kg）")
    save_fig(fig, "dynamics_payload")


# ---------- 实验 ④ 延迟和噪声 ----------
def exp_latency():
    rows, delay_rates, noise_rates = [], [], []
    for d in DELAYS_MS:
        res = [run_trial(0.5, 0.0, seed=s, delay_ms=d) for s in range(REPEATS)]
        delay_rates.append(fall_rate(res))
        rows.append(["动作延迟", f"{d} ms", fall_rate(res), round(mean_speed(res, 2), 3)])
        print(f"  动作延迟 {d:>2} ms  摔倒率 {fall_rate(res):.0%}")
    for sw, sg in NOISE_LEVELS:
        res = [run_trial(0.5, 0.0, seed=s, noise=(sw, sg)) for s in range(REPEATS)]
        noise_rates.append(fall_rate(res))
        rows.append(["IMU噪声", f"角速度{sw} / 重力{sg}", fall_rate(res), round(mean_speed(res, 2), 3)])
        print(f"  IMU 噪声 σ(角速度)={sw:<4} σ(重力)={sg:<4}  摔倒率 {fall_rate(res):.0%}")
    save_csv("latency", ["扰动", "大小", "摔倒率", "实际vx(指令0.5)"], rows)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot(DELAYS_MS, np.array(delay_rates) * 100, "-o", color=COLORS[0], linewidth=2, markersize=8)
    axes[0].set(xlabel="动作延迟 (ms)", ylabel="摔倒率 (%)", ylim=(-5, 105), title="动作延迟的影响")
    labels = [f"{sw}\n{sg}" for sw, sg in NOISE_LEVELS]
    axes[1].plot(range(len(NOISE_LEVELS)), np.array(noise_rates) * 100, "-o", color=COLORS[0], linewidth=2, markersize=8)
    axes[1].set_xticks(range(len(NOISE_LEVELS)), labels)
    axes[1].set(xlabel="IMU 噪声标准差（上：角速度 rad/s，下：重力投影）", ylabel="摔倒率 (%)", ylim=(-5, 105),
                title="IMU 观测噪声的影响")
    for ax in axes:
        style(ax)
    save_fig(fig, "latency")


EXPERIMENTS = {"speed": exp_speed, "push": exp_push, "phase": exp_push_phase,"dynamics": exp_dynamics, "payload": exp_payload_fine, "trace": exp_payload_trace, "latency": exp_latency}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiments", nargs="*", help="要跑的实验：" + " / ".join(EXPERIMENTS) + "，默认全跑")
    parser.add_argument("--quick", action="store_true", help="每组只跑 1 次，用来先检查流程")
    args = parser.parse_args()
    names = args.experiments or list(EXPERIMENTS)
    unknown = set(names) - set(EXPERIMENTS)
    if unknown:
        parser.error(f"未知实验 {unknown}，可选：{list(EXPERIMENTS)}")
    if args.quick:
        REPEATS = PUSH_REPEATS = DYN_REPEATS = 1

    print(f"整机质量 {ROBOT_MASS:.1f} kg，每次仿真 {DURATION} s，每组重复 {REPEATS} 次"
          f"（抗推 {PUSH_REPEATS} 次，摩擦 / 负载 {DYN_REPEATS} 次）")
    t0 = time.time()
    for name in names:
        print(f"\n===== {name} =====")
        EXPERIMENTS[name]()
    print(f"\n完成：共 {n_trials} 次仿真，用时 {time.time() - t0:.0f} s，结果在 {RESULT_DIR}")
