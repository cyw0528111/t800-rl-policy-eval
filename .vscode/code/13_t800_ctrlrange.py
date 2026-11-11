"""加载 T800 的 MuJoCo 模型，遍历每个关节电机（执行器），打印它的 ctrlrange（控制输入范围）。"""
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

    print(f"执行器总数 nu = {model.nu}")
    print(f"{'id':>3}  {pad('电机名称', 28)}{pad('驱动的关节', 24)}ctrlrange")
    for i in range(model.nu):
        actuator = model.actuator(i)
        joint = model.joint(model.actuator_trnid[i, 0])
        # ctrllimited 为 0 时 ctrlrange 不生效，ctrl 不会被限幅
        if model.actuator_ctrllimited[i]:
            low, high = actuator.ctrlrange
            text = f"[{low:>8.2f}, {high:>8.2f}]"
        else:
            text = "不限幅"
        print(f"{i:>3}  {pad(actuator.name, 28)}{pad(joint.name, 24)}{text}")


if __name__ == "__main__":
    main()
