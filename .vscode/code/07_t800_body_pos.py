"""加载 T800 的 MuJoCo 模型，打印每个 body 的 body_pos（相对父 body 的位置偏移）。"""
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

    # body_pos 对应 xml 中 <body pos="..."> 的值，是相对父 body 坐标系的偏移，
    # 与 06 中 data.xpos（世界坐标）不同
    print(f"body 总数: {model.nbody}")
    print(f"{'id':>3}  {pad('名称', 26)}{pad('父 body', 26)}body_pos (m)")
    for i in range(model.nbody):
        name = model.body(i).name or "(未命名)"
        parent = model.body(model.body_parentid[i]).name
        x, y, z = model.body_pos[i]
        print(f"{i:>3}  {pad(name, 26)}{pad(parent, 26)}"
              f"[{x:>7.4f}, {y:>7.4f}, {z:>7.4f}]")


if __name__ == "__main__":
    main()
