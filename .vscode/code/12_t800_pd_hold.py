"""加载 T800 的 MuJoCo 模型，用 PD 控制器让所有关节保持初始位置，并在 viewer 中仿真显示。"""
import os
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

# 工作区根目录（本文件位于 .vscode/code/）
ROOT_DIR = Path(__file__).resolve().parents[2]

# 按关节名关键字分组设置 PD 增益：(kp, kd)，从上到下匹配第一个命中的
GAINS = [
    # 腿部承重，增益要大；踝关节偏软会导致机器人慢慢前倾摔倒
    ("HIP", (2000.0, 30.0)),
    ("KNEE", (2000.0, 30.0)),
    ("ANKLE", (1500.0, 20.0)),
    ("TORSO", (1000.0, 20.0)),
    ("SHOULDER", (300.0, 5.0)),
    ("ELBOW", (200.0, 3.0)),
    ("HEAD", (100.0, 2.0)),
]
PRINT_INTERVAL = 0.5  # 打印间隔（秒，仿真时间）


def build_controller(model):
    """为每个执行器找到对应关节的 qpos / qvel 下标，并确定 kp、kd 和力矩上限。"""
    n = model.nu
    qadr = np.zeros(n, dtype=int)
    dadr = np.zeros(n, dtype=int)
    kp = np.zeros(n)
    kd = np.zeros(n)
    for i in range(n):
        joint = model.joint(model.actuator_trnid[i, 0])
        qadr[i] = joint.qposadr[0]
        dadr[i] = joint.dofadr[0]
        kp[i], kd[i] = next(g for key, g in GAINS if key in joint.name)
    tau_limit = model.actuator_ctrlrange  # 每个电机的力矩范围 [min, max]
    return qadr, dadr, kp, kd, tau_limit


def main():
    # MuJoCo 在 Windows 上无法打开含中文的绝对路径，
    # 所以先切换到根目录，再用纯英文的相对路径加载
    os.chdir(ROOT_DIR)
    model = mujoco.MjModel.from_xml_path("t800.xml")
    data = mujoco.MjData(model)

    qadr, dadr, kp, kd, tau_limit = build_controller(model)
    q_target = data.qpos[qadr].copy()  # 目标 = 初始关节角度
    steps_per_print = max(round(PRINT_INTERVAL / model.opt.timestep), 1)

    print(f"{'t (s)':>6}  {'根部高度 z (m)':>14}  {'最大关节误差 (deg)':>18}")

    # 被动 viewer：仿真循环由我们控制，关闭窗口即退出
    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        while viewer.is_running():
            step_start = time.time()

            # PD 控制：tau = kp * (q目标 - q) - kd * q速度，再按电机力矩范围限幅
            q = data.qpos[qadr]
            dq = data.qvel[dadr]
            tau = kp * (q_target - q) - kd * dq
            data.ctrl[:] = np.clip(tau, tau_limit[:, 0], tau_limit[:, 1])

            mujoco.mj_step(model, data)
            step += 1
            viewer.sync()

            if step % steps_per_print == 0:
                err = np.degrees(np.abs(q_target - data.qpos[qadr]).max())
                print(f"{data.time:>6.2f}  {data.qpos[2]:>14.4f}  {err:>18.3f}")

            # 按 timestep 休眠，让仿真接近实时速度
            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)


if __name__ == "__main__":
    main()
