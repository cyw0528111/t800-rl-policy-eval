"""加载 T800 的 MuJoCo 模型，打印关键维度参数 nq / nv / nu，以及 qpos / qvel / ctrl 的数量和初始内容。"""
import os
from pathlib import Path

import mujoco
import numpy as np

# 工作区根目录（本文件位于 .vscode/code/）
ROOT_DIR = Path(__file__).resolve().parents[2]


def show(name, arr):
    """打印数组的长度和内容（保留 4 位小数，不省略）。"""
    print(f"{name} 数量: {arr.size}")
    print(np.array2string(arr, precision=4, suppress_small=True, max_line_width=100))
    print()


def main():
    # MuJoCo 在 Windows 上无法打开含中文的绝对路径，先切换到根目录再用相对路径加载
    os.chdir(ROOT_DIR)
    model = mujoco.MjModel.from_xml_path("t800.xml")
    data = mujoco.MjData(model)

    # nq: 广义位置维度（自由关节占 7：位置 3 + 四元数 4）
    # nv: 广义速度维度（自由关节占 6：线速度 3 + 角速度 3）
    # nu: 执行器（控制输入）数量
    print(f"nq (广义位置维度) = {model.nq}")
    print(f"nv (广义速度维度) = {model.nv}")
    print(f"nu (执行器数量)   = {model.nu}")
    print()

    # 新建的 MjData 中：qpos 初始化为 model.qpos0，qvel 和 ctrl 初始化为 0
    np.set_printoptions(threshold=np.inf)
    show("qpos", data.qpos)
    show("qvel", data.qvel)
    show("ctrl", data.ctrl)


if __name__ == "__main__":
    main()
