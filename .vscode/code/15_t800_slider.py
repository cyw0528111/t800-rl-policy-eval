"""用 viewer 右侧 Control 面板的滑块，手动拖动 T800 的 25 个关节。"""
import os
import time
from pathlib import Path

import mujoco
import mujoco.viewer

ROOT_DIR = Path(__file__).resolve().parents[2]
KP, KD = 300.0, 5.0  # 位置伺服的刚度和阻尼

os.chdir(ROOT_DIR)  # MuJoCo 无法打开含中文的绝对路径
model = mujoco.MjModel.from_xml_path("t800.xml")
data = mujoco.MjData(model)

# 把力矩电机改成位置伺服：力矩 = KP * (ctrl - q) - KD * dq，滑块值就是目标角度
for i in range(model.nu):
    joint_id = model.actuator_trnid[i, 0]
    model.actuator_forcerange[i] = model.actuator_ctrlrange[i]  # 原来的力矩上限
    model.actuator_forcelimited[i] = 1
    model.actuator_ctrlrange[i] = model.jnt_range[joint_id]     # 滑块范围 = 关节范围
    model.actuator_gainprm[i, 0] = KP
    model.actuator_biasprm[i, :3] = [0.0, -KP, -KD]
    model.actuator_biastype[i] = mujoco.mjtBias.mjBIAS_AFFINE

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)  # 粗略控制成实时速度
