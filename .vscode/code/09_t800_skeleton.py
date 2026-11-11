"""加载 T800 的 MuJoCo 模型，用 matplotlib 3D 绘制机器人骨架（body 世界坐标 + 父子连线）。"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

# 工作区根目录（本文件位于 .vscode/code/）
ROOT_DIR = Path(__file__).resolve().parents[2]

# 解决中文乱码：指定 Windows 自带的中文字体，并让负号正常显示
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    # MuJoCo 在 Windows 上无法打开含中文的绝对路径，先切换到根目录再用相对路径加载
    os.chdir(ROOT_DIR)
    model = mujoco.MjModel.from_xml_path("t800.xml")
    data = mujoco.MjData(model)

    # 先做一次正运动学，得到各 body 的世界坐标 xpos
    mujoco.mj_forward(model, data)
    xpos = data.xpos

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(projection="3d")

    # 骨骼：每个 body 与其父 body 连线（跳过 world，id=0）
    for i in range(1, model.nbody):
        parent = model.body_parentid[i]
        if parent == 0:
            continue
        p, c = xpos[parent], xpos[i]
        ax.plot([p[0], c[0]], [p[1], c[1]], [p[2], c[2]], color="#4a6fa5", linewidth=2)

    # 关节点：body 原点
    pts = xpos[1:]
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color="#d1495b", s=25, depthshade=False)

    # 标注 body 名称
    for i in range(1, model.nbody):
        name = model.body(i).name or f"body{i}"
        x, y, z = xpos[i]
        ax.text(x, y, z, f" {name}", fontsize=7, color="#333333")

    # 三轴等比例，避免骨架被拉伸变形
    mins, maxs = pts.min(axis=0), pts.max(axis=0)
    center = (mins + maxs) / 2
    half = (maxs - mins).max() / 2 + 0.05
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1, 1, 1))

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"T800 机器人骨架结构（共 {model.nbody - 1} 个 body）")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
