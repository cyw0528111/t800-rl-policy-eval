"""给 T800 每个电机输入 10000 的控制值，打印真正作用到关节上的力矩，观察 ctrlrange 限幅（clamp）是否生效。"""
import os
import unicodedata
from pathlib import Path

import mujoco

# 工作区根目录（本文件位于 .vscode/code/）
ROOT_DIR = Path(__file__).resolve().parents[2]

CTRL_INPUT = 10000.0  # 故意远超 ctrlrange 的控制输入


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

    # 限幅开关：clampctrl 被禁用时，ctrlrange 不会生效
    clamp_disabled = model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL
    print(f"clampctrl 限幅开关: {'已禁用' if clamp_disabled else '已启用'}")

    # 写入超大控制值，再做一次前向计算，让 MuJoCo 算出执行器实际输出
    data.ctrl[:] = CTRL_INPUT
    mujoco.mj_forward(model, data)

    # 注意：MuJoCo 不会修改 data.ctrl 本身，限幅发生在内部计算里，
    # 真正输出的力矩要看 data.actuator_force
    print(f"{'id':>3}  {pad('电机名称', 28)}{'ctrl 输入':>10}  {'ctrlrange':>20}  {'实际力矩':>10}  生效")
    for i in range(model.nu):
        name = model.actuator(i).name
        low, high = model.actuator_ctrlrange[i]
        force = data.actuator_force[i]
        ok = "是" if abs(force - high) < 1e-6 else "否"
        print(f"{i:>3}  {pad(name, 28)}{data.ctrl[i]:>11.1f}  [{low:>8.1f}, {high:>8.1f}]"
              f"  {force:>12.1f}  {ok}")


if __name__ == "__main__":
    main()
