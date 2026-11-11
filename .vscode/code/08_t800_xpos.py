"""加载 T800 的 MuJoCo 模型，打印每个 body 的 xpos（世界坐标系下的位置）。"""
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

    # xpos 是仿真计算结果，刚创建的 data 里全是 0，
    # 必须先做一次正运动学（mj_forward）才能得到各 body 的世界坐标
    mujoco.mj_forward(model, data)

    print(f"body 总数: {model.nbody}")
    print(f"{'id':>3}  {pad('名称', 26)}xpos (m)")
    for i in range(model.nbody):
        name = model.body(i).name or "(未命名)"
        x, y, z = data.xpos[i]
        print(f"{i:>3}  {pad(name, 26)}[{x:>7.4f}, {y:>7.4f}, {z:>7.4f}]")


if __name__ == "__main__":
    main()
