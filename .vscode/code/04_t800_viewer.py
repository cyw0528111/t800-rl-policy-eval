"""加载 T800 的 MuJoCo 模型，并在 viewer 窗口中实时仿真显示。"""
import os
import time
from pathlib import Path

import mujoco
import mujoco.viewer

# 工作区根目录（本文件位于 .vscode/code/）
ROOT_DIR = Path(__file__).resolve().parents[2]


def main():
    # MuJoCo 在 Windows 上无法打开含中文的绝对路径，
    # 所以先切换到根目录，再用纯英文的相对路径加载
    os.chdir(ROOT_DIR)
    model = mujoco.MjModel.from_xml_path("t800.xml")
    data = mujoco.MjData(model)

    # 被动 viewer：仿真循环由我们控制，关闭窗口即退出
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()

            # 按 timestep 休眠，让仿真接近实时速度
            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)


if __name__ == "__main__":
    main()
