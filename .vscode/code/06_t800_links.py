"""加载 T800 的 MuJoCo 模型，打印所有连杆（body）的名称、父连杆、质量和位置。"""
import os
import unicodedata
from pathlib import Path

import mujoco

# 工作区根目录（本文件位于 .vscode/code/）
ROOT_DIR = Path(__file__).resolve().parents[2]


def pad(text, width):
    """按显示宽度左对齐填充：中文等全角字符占 2 格。"""
    display = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)
    return text + " " * max(width - display, 0)


def main():
    # MuJoCo 在 Windows 上无法打开含中文的绝对路径，
    # 所以先切换到根目录，再用纯英文的相对路径加载
    os.chdir(ROOT_DIR)
    model = mujoco.MjModel.from_xml_path("t800.xml")
    data = mujoco.MjData(model)
    # 计算正运动学，得到初始姿态下各连杆的世界坐标
    mujoco.mj_forward(model, data)

    print(f"连杆总数: {model.nbody}（含 world）")
    print(f"总质量: {sum(model.body_mass):.3f} kg")
    print(f"{'id':>3}  {pad('名称', 26)}{pad('父连杆', 26)}"
          f"{pad('质量(kg)', 10)}世界坐标 (m)")
    for i in range(model.nbody):
        body = model.body(i)
        parent = model.body(body.parentid[0]).name
        x, y, z = data.xpos[i]
        print(f"{i:>3}  {pad(body.name or '(未命名)', 26)}{pad(parent, 26)}"
              f"{body.mass[0]:<10.3f}[{x:.3f}, {y:.3f}, {z:.3f}]")


if __name__ == "__main__":
    main()
