"""加载 T800 的 MuJoCo 模型，在 viewer 中仿真，每隔 100 ms 打印一次机器人根部的 pos 和 rpy，共 10 秒。"""
import os
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

# 工作区根目录（本文件位于 .vscode/code/）
ROOT_DIR = Path(__file__).resolve().parents[2]

PRINT_INTERVAL = 0.1  # 打印间隔（秒，仿真时间）
DURATION = 10.0       # 总仿真时长（秒）


def quat_to_rpy(quat):
    """MuJoCo 四元数 [w, x, y, z] 转 roll / pitch / yaw（弧度，ZYX 顺序）。"""
    w, x, y, z = quat
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([roll, pitch, yaw])


def print_state(data):
    """打印根部自由关节的位置和姿态：qpos[0:3] 为位置，qpos[3:7] 为四元数。"""
    x, y, z = data.qpos[0:3]
    roll, pitch, yaw = np.degrees(quat_to_rpy(data.qpos[3:7]))
    print(f"{data.time:>6.2f}  [{x:>8.4f}, {y:>8.4f}, {z:>8.4f}]"
          f"  [{roll:>8.2f}, {pitch:>8.2f}, {yaw:>8.2f}]")


def main():
    # MuJoCo 在 Windows 上无法打开含中文的绝对路径，
    # 所以先切换到根目录，再用纯英文的相对路径加载
    os.chdir(ROOT_DIR)
    model = mujoco.MjModel.from_xml_path("t800.xml")
    data = mujoco.MjData(model)

    # 每隔多少步打印一次（按仿真时间计，不受电脑快慢影响）
    steps_per_print = max(round(PRINT_INTERVAL / model.opt.timestep), 1)
    total_steps = round(DURATION / model.opt.timestep)

    print(f"timestep = {model.opt.timestep} s，每 {steps_per_print} 步打印一次")
    print(f"{'t (s)':>6}  {'pos (m)':^30}  {'rpy (deg)':^30}")
    print_state(data)

    # 被动 viewer：仿真循环由我们控制，满 10 秒或关闭窗口即退出
    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        while viewer.is_running() and step < total_steps:
            step_start = time.time()
            mujoco.mj_step(model, data)
            step += 1
            viewer.sync()

            if step % steps_per_print == 0:
                print_state(data)

            # 按 timestep 休眠，让仿真接近实时速度
            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)


if __name__ == "__main__":
    main()
