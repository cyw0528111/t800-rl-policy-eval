"""按 EngineAI SDK 的真实部署参数，用 MNN 策略控制 T800 稳定行走。

参数来源：engineai_robotics_native_sdk-main
  assets/config/t800/rl_walking_example/default.yaml
  src/runner/rl_walking_example/src/rl_walking_example_runner.cc

obs 1083 = 15 帧历史（旧 -> 新）× 72 + 指令 3
  每帧 72 = (q - q默认) 22 + dq*0.05 22 + 上一帧动作 22 + 角速度 3 + 重力投影 3
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

# ---------- 速度指令 ----------
VX, VY = 0.5, 0.0      # 前进 / 侧向速度 (m/s)
YAW_RATE = 0.0         # 转向速度 (rad/s)，0 表示保持当前朝向

# ---------- SDK 部署参数（25 个关节，全身顺序） ----------
DEFAULT_Q = np.array([-0.06, 0, 0, 0.12, -0.06, 0,  -0.06, 0, 0, 0.12, -0.06, 0,  0,
                      0, 0.15, 0, -0.25, 0,  0, -0.15, 0, -0.25, 0,  0, 0])
KP = np.array([180, 100, 100, 180, 40, 40,  180, 100, 100, 180, 40, 40,  100,
               60, 50, 50, 60, 50,  60, 50, 50, 60, 50,  100, 100], float)
KD = np.array([5, 3, 3, 5, 0.3, 0.3,  5, 3, 3, 5, 0.3, 0.3,  5,
               0.3, 0.3, 0.3, 0.3, 0.3,  0.3, 0.3, 0.3, 0.3, 0.3,  1, 1], float)
SOFT_TORQUE = np.array([0.8] * 12 + [0.9] * 13)          # 软力矩上限系数
POLICY = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]  # 22 个策略关节
ACTION_SCALE = np.array([0.5, 0.2, 0.2, 0.5, 0.5, 0.2,  0.5, 0.2, 0.2, 0.5, 0.5, 0.2,
                         0.2, 0.2, 0.05, 0.2, 0.05,  0.2, 0.2, 0.05, 0.2, 0.05])
DQ_SCALE, CMD_SCALE = 0.05, 2.0
YAW_KP = 2.0           # 航向 PD：obs 指令第 3 维 = clip(2 * (目标 yaw - 当前 yaw), -1, 1)
HISTORY, CLIP = 15, 100.0
DECIMATION = 5         # 0.002 s × 5 = 100 Hz 推理
LOG_INTERVAL = 0.1     # 日志间隔 (s)

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


def quat_to_rpy(q):
    """MuJoCo 四元数 [w, x, y, z] 转 roll / pitch / yaw（弧度）。"""
    w, x, y, z = q
    return np.array([np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
                     np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)),
                     np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))])


# 每个执行器对应关节的 qpos / qvel 下标（执行器顺序就是 SDK 的全身顺序）
qadr = np.array([model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in range(model.nu)])
dadr = np.array([model.jnt_dofadr[model.actuator_trnid[i, 0]] for i in range(model.nu)])
tau_limit = model.actuator_ctrlrange[:, 1] * SOFT_TORQUE

# 从默认站姿开始
data.qpos[qadr] = DEFAULT_Q
mujoco.mj_forward(model, data)

history = None
action = np.zeros(22, np.float32)
q_des = DEFAULT_Q.copy()
desired_yaw = quat_to_rpy(data.qpos[3:7])[2]
step = 0
log_every = round(LOG_INTERVAL / model.opt.timestep)

print(f"速度指令 vx={VX} vy={VY} yaw_rate={YAW_RATE}，推理 {1 / (model.opt.timestep * DECIMATION):.0f} Hz")
print(f"{'t':>5} | {'pos x / y / z (m)':^22} | {'rpy (deg)':^20} | {'|a|max':>6} | {'|tau|max':>8} | 脚底力 L / R (N)")

FOOT_L = {model.body("LINK_ANKLE_ROLL_L").id, model.body("LINK_FOOT_L").id}
FOOT_R = {model.body("LINK_ANKLE_ROLL_R").id, model.body("LINK_FOOT_R").id}


def foot_force(foot_bodies):
    """这只脚所有接触点的法向力之和 (N)，为 0 表示悬空。"""
    total, f6 = 0.0, np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        if model.geom_bodyid[c.geom1] in foot_bodies or model.geom_bodyid[c.geom2] in foot_bodies:
            mujoco.mj_contactForce(model, data, i, f6)
            total += abs(f6[0])
    return total


with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        if step % DECIMATION == 0:
            # 1) 拼一帧 72 维观测
            R = data.xmat[1].reshape(3, 3)                 # 根部姿态
            gravity = R.T @ [0, 0, -1]                     # 重力在机身坐标系下的投影
            q, dq = data.qpos[qadr], data.qvel[dadr]
            frame = np.concatenate([(q - DEFAULT_Q)[POLICY], dq[POLICY] * DQ_SCALE, action,
                                    data.qvel[3:6], gravity])
            frame = np.clip(frame, -CLIP, CLIP).astype(np.float32)
            history = [frame] * HISTORY if history is None else history[1:] + [frame]

            # 2) 指令：vx、vy 乘 2；第 3 维是航向 PD，偏离目标朝向就往回转
            desired_yaw += YAW_RATE * model.opt.timestep * DECIMATION
            yaw_err = np.angle(np.exp(1j * (desired_yaw - quat_to_rpy(data.qpos[3:7])[2])))
            cmd = np.array([VX * CMD_SCALE, VY * CMD_SCALE, np.clip(YAW_KP * yaw_err, -1, 1)], np.float32)

            # 3) 推理，得到目标关节角
            action = np.clip(infer(np.concatenate(history + [cmd])), -CLIP, CLIP)
            q_des = DEFAULT_Q.copy()                       # 腰、头保持默认角
            q_des[POLICY] += action * ACTION_SCALE

        # 4) PD 力矩，按软力矩上限限幅
        tau = KP * (q_des - data.qpos[qadr]) - KD * data.qvel[dadr]
        data.ctrl[:] = np.clip(tau, -tau_limit, tau_limit)

        mujoco.mj_step(model, data)
        step += 1
        viewer.sync()

        if step % log_every == 0:
            x, y, z = data.qpos[0:3]
            roll, pitch, yaw = np.degrees(quat_to_rpy(data.qpos[3:7]))
            print(f"{data.time:>5.2f} | {x:>6.3f} {y:>6.3f} {z:>6.3f}    | {roll:>6.1f} {pitch:>6.1f} {yaw:>6.1f} | "
                  f"{np.abs(action).max():>6.2f} | {np.abs(data.ctrl).max():>8.1f} | "
                  f"{foot_force(FOOT_L):>7.1f} / {foot_force(FOOT_R):>7.1f}")

        time.sleep(model.opt.timestep)  # 粗略控制成实时速度
