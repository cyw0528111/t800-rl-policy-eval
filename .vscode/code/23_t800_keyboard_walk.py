"""在 20 的基础上加键盘控制：在 viewer 窗口里按键，控制 T800 前后左右走。

按键（每按一次改变 0.1 m/s）：
  ↑ / W  前进加速      ↓ / S  后退（减速）
  ← / A  向左平移      → / D  向右平移
注意：MuJoCo viewer 自带快捷键，W 会切换线框显示、S 会切换阴影，
      所以推荐用方向键；A / D 的副作用（Auto Connect / Static Body 显示）代码里会自动还原。
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

# ---------- 键盘控制 ----------
STEP = 0.1                       # 每按一次的速度变化 (m/s)
VX_RANGE = (-0.5, 1.0)           # 前后速度范围
VY_RANGE = (-0.4, 0.4)           # 左右速度范围
KEYS = {265: (STEP, 0), 87: (STEP, 0),      # ↑ / W
        264: (-STEP, 0), 83: (-STEP, 0),    # ↓ / S
        263: (0, STEP), 65: (0, STEP),      # ← / A（+y 是机器人左侧）
        262: (0, -STEP), 68: (0, -STEP)}    # → / D
cmd_vel = [0.0, 0.0]             # 当前速度指令 [vx, vy]，开始时原地站立


def on_key(keycode):
    """viewer 按键回调：按一次，速度指令改变一档。"""
    if keycode in KEYS:
        dvx, dvy = KEYS[keycode]
        cmd_vel[0] = round(float(np.clip(cmd_vel[0] + dvx, *VX_RANGE)), 2)
        cmd_vel[1] = round(float(np.clip(cmd_vel[1] + dvy, *VY_RANGE)), 2)
        print(f"速度指令 vx = {cmd_vel[0]:+.1f} m/s, vy = {cmd_vel[1]:+.1f} m/s")


# ---------- SDK 部署参数（25 个关节，全身顺序，同 20） ----------
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

data.qpos[qadr] = DEFAULT_Q
mujoco.mj_forward(model, data)
history, action, q_des = None, np.zeros(22, np.float32), DEFAULT_Q.copy()
desired_yaw = yaw_of(data.qpos[3:7])  # 保持出发时的朝向
step = 0
FLAG_IDS = [mujoco.mjtVisFlag.mjVIS_AUTOCONNECT, mujoco.mjtVisFlag.mjVIS_STATIC]  # A / D 会切换的显示开关

print(__doc__)
with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
    flags0 = [viewer.opt.flags[i] for i in FLAG_IDS]
    while viewer.is_running():
        if step % DECIMATION == 0:
            R = data.xmat[1].reshape(3, 3)
            q, dq = data.qpos[qadr], data.qvel[dadr]
            frame = np.concatenate([(q - DEFAULT_Q)[POLICY], dq[POLICY] * DQ_SCALE, action,
                                    data.qvel[3:6], R.T @ [0, 0, -1]])
            frame = np.clip(frame, -CLIP, CLIP).astype(np.float32)
            history = [frame] * HISTORY if history is None else history[1:] + [frame]
            yaw_err = np.angle(np.exp(1j * (desired_yaw - yaw_of(data.qpos[3:7]))))
            cmd = np.array([cmd_vel[0] * CMD_SCALE, cmd_vel[1] * CMD_SCALE,
                            np.clip(YAW_KP * yaw_err, -1, 1)], np.float32)
            action = np.clip(infer(np.concatenate(history + [cmd])), -CLIP, CLIP)
            q_des = DEFAULT_Q.copy()
            q_des[POLICY] += action * ACTION_SCALE

        tau = KP * (q_des - data.qpos[qadr]) - KD * data.qvel[dadr]
        data.ctrl[:] = np.clip(tau, -tau_limit, tau_limit)
        mujoco.mj_step(model, data)
        step += 1

        if step % 50 == 0:  # 每 0.1 s 刷新一次界面上的文字，并还原被 A / D 切换的显示开关
            with viewer.lock():
                for i, f in zip(FLAG_IDS, flags0):
                    viewer.opt.flags[i] = f
            viewer.set_texts((None, None, "vx / vy (m/s)", f"{cmd_vel[0]:+.1f} / {cmd_vel[1]:+.1f}"))
        viewer.sync()
        time.sleep(model.opt.timestep)
