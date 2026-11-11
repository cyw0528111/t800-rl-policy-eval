"""用 MNN 神经网络策略控制 T800 向前走：观测 -> 推理 -> 目标角 -> PD 力矩。

注意：下面"需要和训练配置对齐"的参数目前是根据维度推测的，
拿到训练配置后替换成真实值，机器人才能真正走起来。
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

# ---------- 需要和训练配置对齐 ----------
CMD = [0.5, 0.0, 0.0]   # 速度指令 [vx, vy, yaw_rate]
HISTORY = 15            # 历史帧数，1083 = 72 * 15 + 3
ACTION_SCALE = 0.5      # 目标角 = 默认角(0) + action * ACTION_SCALE
KP, KD = 200.0, 5.0     # 底层 PD 增益
DECIMATION = 10         # 每 10 个仿真步推理一次（0.002 s * 10 = 50 Hz）
# ---------------------------------------

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


# 策略控制的 22 个关节：去掉腰和头（推测）
ids = [i for i in range(model.nu) if not any(k in model.actuator(i).name for k in ("TORSO", "HEAD"))]
qadr = [model.joint(model.actuator_trnid[i, 0]).qposadr[0] for i in ids]
dadr = [model.joint(model.actuator_trnid[i, 0]).dofadr[0] for i in ids]

history = [np.zeros(72, np.float32)] * HISTORY
action = np.zeros(22, np.float32)
step = 0

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        if step % DECIMATION == 0:
            # 单帧观测 72 维：角速度 3 + 重力投影 3 + 关节角 22 + 关节速度 22 + 上一帧动作 22
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, data.qpos[3:7])
            gravity = R.reshape(3, 3).T @ [0, 0, -1]
            frame = np.concatenate([data.qvel[3:6], gravity, data.qpos[qadr], data.qvel[dadr], action])
            history = history[1:] + [frame.astype(np.float32)]
            obs = np.concatenate(history + [np.array(CMD, np.float32)])
            action = infer(obs)

        # PD：把网络输出的目标角变成力矩
        target = action * ACTION_SCALE
        tau = KP * (target - data.qpos[qadr]) - KD * data.qvel[dadr]
        data.ctrl[:] = 0
        data.ctrl[ids] = tau  # 超出 ctrlrange 的部分由 MuJoCo 自动限幅

        mujoco.mj_step(model, data)
        step += 1
        viewer.sync()
        time.sleep(model.opt.timestep)  # 粗略控制成实时速度
