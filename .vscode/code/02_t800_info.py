"""加载 T800 的 MuJoCo 模型，打印身体部件、关节和电机数量。"""
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

    # nbody 包含 world body（id 0）
    print(f"身体部件数量 (nbody): {model.nbody}")
    print(f"关节数量 (njnt): {model.njnt}")
    print(f"电机数量 (nu): {model.nu}")


if __name__ == "__main__":
    main()
