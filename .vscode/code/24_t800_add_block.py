"""不修改 t800.xml，用 MjSpec 在加载时动态加入一个红色木块（边长 8 cm 的正方体）。

做法：MjSpec.from_file 读入 t800.xml -> 在 worldbody 下加一个带自由关节的方块 -> compile 成新的 model。
机器人用大增益 PD 保持初始姿态站着（同 12），木块从机器人前方 0.5 m 高处落到地上。
"""
import os
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]

BLOCK_SIZE = 0.08              # 边长 (m)
BLOCK_POS = [0.6, 0.0, 0.5]    # 初始位置：机器人前方 0.6 m，离地 0.5 m
WOOD_DENSITY = 600             # 木头密度 (kg/m³)，质量 ≈ 0.08³ × 600 ≈ 0.31 kg

# ---------- 1) 动态加木块 ----------
os.chdir(ROOT_DIR)  # MuJoCo 无法打开含中文的绝对路径
spec = mujoco.MjSpec.from_file("t800.xml")
block = spec.worldbody.add_body(name="red_block", pos=BLOCK_POS)
block.add_freejoint()                       # 自由关节：木块可以自由移动、翻滚
block.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
               size=[BLOCK_SIZE / 2] * 3,   # box 的 size 是半边长
               rgba=[0.9, 0.1, 0.1, 1.0],   # 红色
               density=WOOD_DENSITY)
model = spec.compile()                      # t800.xml 文件本身没有被修改
data = mujoco.MjData(model)

block_id = model.body("red_block").id
print(f"木块已加入：body id = {block_id}，质量 = {model.body_mass[block_id]:.3f} kg")

# ---------- 2) 机器人 PD 保持初始姿态（增益同 12） ----------
GAINS = [("HIP", (2000.0, 30.0)), ("KNEE", (2000.0, 30.0)), ("ANKLE", (1500.0, 20.0)),
         ("TORSO", (1000.0, 20.0)), ("SHOULDER", (300.0, 5.0)), ("ELBOW", (200.0, 3.0)), ("HEAD", (100.0, 2.0))]
joints = [model.joint(model.actuator_trnid[i, 0]) for i in range(model.nu)]
qadr = np.array([j.qposadr[0] for j in joints])
dadr = np.array([j.dofadr[0] for j in joints])
kp = np.array([next(g for k, g in GAINS if k in j.name)[0] for j in joints])
kd = np.array([next(g for k, g in GAINS if k in j.name)[1] for j in joints])
q_target = data.qpos[qadr].copy()
lim = model.actuator_ctrlrange

with mujoco.viewer.launch_passive(model, data) as viewer:
    step = 0
    while viewer.is_running():
        tau = kp * (q_target - data.qpos[qadr]) - kd * data.qvel[dadr]
        data.ctrl[:] = np.clip(tau, lim[:, 0], lim[:, 1])
        mujoco.mj_step(model, data)
        step += 1
        if step % 250 == 0:  # 每 0.5 s 打印一次木块位置
            x, y, z = data.xpos[block_id]
            print(f"t = {data.time:5.2f} s   木块位置 [{x:.3f}, {y:.3f}, {z:.3f}]")
        viewer.sync()
        time.sleep(model.opt.timestep)
