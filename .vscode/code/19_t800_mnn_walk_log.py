"""在 18 的基础上修正 obs 排布并加日志：每 100 ms 打印机器人 pos / rpy、推理信息、PD 信息，方便分析步态。

obs 排布（对网络逐维扰动做灵敏度分析反推出来的）：
  每帧 72 维 = 关节角偏差 22 + 关节速度 22 + 上一帧动作 22 + 角速度 3 + 重力投影 3
  obs 1083 维 = 15 帧历史（旧 -> 新） + 速度指令 3

日志各列含义：
  t        仿真时间 (s)
  pos      根部位置 x / y / z (m)
  rpy      根部姿态 roll / pitch / yaw (deg)
  推理     本周期内平均推理耗时 (ms)，|action| 的最大值
  PD误差   |目标角 - 当前角| 的最大值 (deg) 及对应关节
  力矩     |tau| 的最大值 (N·m)，超出 ctrlrange 被限幅的关节数
  脚底力   左 / 右脚接触法向力 (N)，为 0 表示该脚悬空，用来看步态
"""
import os
import time
from pathlib import Path

import MNN
import mujoco
import mujoco.viewer
import numpy as np

CODE_DIR = Path(__file__).resolve().parent
ROOT_DIR = CODE_DIR.parents[1]

# ---------- 需要和训练配置对齐（目前是在仿真里搜索出来的最佳值） ----------
CMD = [0.5, 0.0]               # 速度指令 [vx, vy] (m/s)
CMD_SCALE = 0.5                # 速度指令进 obs 前的缩放：obs = CMD * 0.5
HEADING_K = 0.25               # 航向闭环：yaw_rate 指令 = -HEADING_K * 当前 yaw，让机器人保持朝 +x 走
HISTORY = 15                   # 历史帧数
DEFAULT_POSE = {"HIP_PITCH": -0.3, "KNEE": 0.6, "ANKLE_PITCH": -0.3}  # 默认站姿（屈膝），其余关节为 0
ACTION_SCALE = 0.25            # 目标角 = 默认角 + action * ACTION_SCALE
ACTION_CLIP = 18.0             # 动作限幅，防止数值发散
DQ_SCALE = 0.05                # 关节速度观测缩放
ANG_VEL_SCALE = 1.0            # 角速度观测缩放
GAINS = [("HIP", (300.0, 5.7)), ("KNEE", (300.0, 5.7)), ("ANKLE", (80.0, 4.5)),
         ("SHOULDER", (80.0, 1.4)), ("ELBOW", (80.0, 1.4))]  # 策略关节的 PD (kp, kd)
HOLD_KP, HOLD_KD = 100.0, 3.0  # 腰、头不归策略控制，用 PD 锁在 0
DECIMATION = 10                # 每 10 个仿真步推理一次（0.002 s * 10 = 50 Hz）
# -------------------------------------------------------------------------
LOG_INTERVAL = 0.1             # 日志间隔（秒，仿真时间）

# 加载模型（MuJoCo 和 MNN 都打不开含中文的绝对路径，所以先切目录再用相对路径）
os.chdir(ROOT_DIR)
model = mujoco.MjModel.from_xml_path("t800.xml")
data = mujoco.MjData(model)
os.chdir(CODE_DIR / "MNN-CODE")
net = MNN.Interpreter("policy/t800_260318_150533_60000.mnn")
session = net.createSession()


def infer(obs):
    """把 obs 送进网络，返回 22 维动作。"""
    x = MNN.Tensor((1, obs.size), MNN.Halide_Type_Float, obs.reshape(1, -1), MNN.Tensor_DimensionType_Caffe)
    net.getSessionInput(session).copyFrom(x)
    net.runSession(session)
    y = MNN.Tensor((1, 22), MNN.Halide_Type_Float, np.zeros((1, 22), np.float32), MNN.Tensor_DimensionType_Caffe)
    net.getSessionOutput(session).copyToHostTensor(y)
    return np.array(y.getData(), np.float32)


def quat_to_rpy(quat):
    """MuJoCo 四元数 [w, x, y, z] 转 roll / pitch / yaw（度，ZYX 顺序）。"""
    w, x, y, z = quat
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.degrees([roll, pitch, yaw])


def foot_force(foot_bodies):
    """统计所有接触点中，属于这只脚的接触法向力之和 (N)。"""
    total = 0.0
    f6 = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        if model.geom_bodyid[c.geom1] in foot_bodies or model.geom_bodyid[c.geom2] in foot_bodies:
            mujoco.mj_contactForce(model, data, i, f6)
            total += abs(f6[0])  # f6[0] 是接触法向力
    return total


# 左右脚：踝 roll 连杆和脚连杆都算作脚
FOOT_L = {model.body("LINK_ANKLE_ROLL_L").id, model.body("LINK_FOOT_L").id}
FOOT_R = {model.body("LINK_ANKLE_ROLL_R").id, model.body("LINK_FOOT_R").id}

# 策略控制的 22 个关节：去掉腰和头；腰和头单独用 PD 锁住
ids = [i for i in range(model.nu) if not any(k in model.actuator(i).name for k in ("TORSO", "HEAD"))]
hold = [i for i in range(model.nu) if i not in ids]
qadr = [model.joint(model.actuator_trnid[i, 0]).qposadr[0] for i in ids]
dadr = [model.joint(model.actuator_trnid[i, 0]).dofadr[0] for i in ids]
hold_qadr = [model.joint(model.actuator_trnid[i, 0]).qposadr[0] for i in hold]
hold_dadr = [model.joint(model.actuator_trnid[i, 0]).dofadr[0] for i in hold]
names = [model.actuator(i).name.replace("motor_", "") for i in ids]
tau_limit = model.actuator_ctrlrange[ids]
joint_range = model.jnt_range[[model.actuator_trnid[i, 0] for i in ids]]

q_default = np.array([next((v for k, v in DEFAULT_POSE.items() if k in n), 0.0) for n in names])
kp = np.array([next(g for k, g in GAINS if k in n)[0] for n in names])
kd = np.array([next(g for k, g in GAINS if k in n)[1] for n in names])

# 从默认站姿开始
data.qpos[qadr] = q_default
mujoco.mj_forward(model, data)

history = None
action = np.zeros(22, np.float32)
target = q_default.copy()
step = 0
log_every = round(LOG_INTERVAL / model.opt.timestep)
infer_times = []  # 两次日志之间每次推理的耗时 (ms)

print(f"策略控制 {len(ids)} 个关节，推理频率 {1 / (model.opt.timestep * DECIMATION):.0f} Hz，"
      f"日志间隔 {LOG_INTERVAL * 1000:.0f} ms，速度指令 {CMD}")
print(f"{'t':>5} | {'pos x / y / z (m)':^22} | {'rpy (deg)':^22} | {'推理 ms':>7} {'|a|max':>6} | "
      f"{'PD误差max(deg) 关节':^28} | {'|tau|max':>8} {'限幅':>4} | {'脚底力 L / R (N)':^16}")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        if step % DECIMATION == 0:
            # 单帧观测 72 维：关节角偏差 22 + 关节速度 22 + 上一帧动作 22 + 角速度 3 + 重力投影 3
            R = data.xmat[1].reshape(3, 3)  # 根部姿态（body 1 = LINK_BASE）
            gravity = R.T @ [0, 0, -1]
            frame = np.concatenate([data.qpos[qadr] - q_default, data.qvel[dadr] * DQ_SCALE, action,
                                    data.qvel[3:6] * ANG_VEL_SCALE, gravity]).astype(np.float32)
            if history is None:
                history = [frame] * HISTORY  # 第一帧用当前观测填满历史
            history = history[1:] + [frame]
            yaw = np.radians(quat_to_rpy(data.qpos[3:7])[2])
            cmd = np.array([CMD[0] * CMD_SCALE, CMD[1] * CMD_SCALE,
                            np.clip(-HEADING_K * yaw, -1.0, 1.0)], np.float32)
            obs = np.concatenate(history + [cmd])
            t0 = time.perf_counter()
            action = np.clip(infer(obs), -ACTION_CLIP, ACTION_CLIP)
            infer_times.append((time.perf_counter() - t0) * 1000)
            # 目标角限制在关节活动范围内，避免踝关节追一个够不到的角度
            target = np.clip(q_default + action * ACTION_SCALE, joint_range[:, 0], joint_range[:, 1])

        # PD：策略关节跟踪网络给的目标角，腰和头锁在 0
        err = target - data.qpos[qadr]
        tau = kp * err - kd * data.qvel[dadr]
        data.ctrl[ids] = np.clip(tau, tau_limit[:, 0], tau_limit[:, 1])
        data.ctrl[hold] = HOLD_KP * (0 - data.qpos[hold_qadr]) - HOLD_KD * data.qvel[hold_dadr]

        mujoco.mj_step(model, data)
        step += 1
        viewer.sync()

        if step % log_every == 0:
            x, y, z = data.qpos[0:3]
            roll, pitch, yaw = quat_to_rpy(data.qpos[3:7])
            worst = int(np.argmax(np.abs(err)))
            n_clip = int(np.sum((tau < tau_limit[:, 0]) | (tau > tau_limit[:, 1])))
            print(f"{data.time:>5.2f} | {x:>6.3f} {y:>6.3f} {z:>6.3f}    | "
                  f"{roll:>6.1f} {pitch:>6.1f} {yaw:>6.1f}    | "
                  f"{np.mean(infer_times):>7.2f} {np.abs(action).max():>6.2f} | "
                  f"{np.degrees(abs(err[worst])):>7.2f}  {names[worst]:<19} | "
                  f"{np.abs(tau).max():>8.1f} {n_clip:>4} | "
                  f"{foot_force(FOOT_L):>7.1f} / {foot_force(FOOT_R):>7.1f}")
            infer_times.clear()

        time.sleep(model.opt.timestep)  # 粗略控制成实时速度
