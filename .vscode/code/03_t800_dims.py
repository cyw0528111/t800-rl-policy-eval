"""加载 T800 的 MuJoCo 模型，打印 nq、nv、nu 三个维度参数。"""
import os
from pathlib import Path

import mujoco

# 工作区根目录（本文件位于 .vscode/code/）
ROOT_DIR = Path(__file__).resolve().parents[2]


def main():
    # MuJoCo 在 Windows 上无法打开含中文的绝对路径，
    # 所以先切换到根目录，再用纯英文的相对路径加载
    os.chdir(ROOT_DIR)
    model = mujoco.MjModel.from_xml_path("t800.xml")

    # nq：广义位置维度（freejoint 占 7：3 位置 + 4 四元数）
    print(f"nq (广义位置维度): {model.nq}")
    # nv：广义速度维度（freejoint 占 6：3 线速度 + 3 角速度）
    print(f"nv (广义速度维度): {model.nv}")
    # nu：执行器（电机）数量
    print(f"nu (执行器数量): {model.nu}")


if __name__ == "__main__":
    main()
