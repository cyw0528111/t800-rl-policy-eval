"""扫描 T800 的初始高度：从 0.5 m 到 1.5 m，每次加 2 cm，统计机器人和地面的接触点数量。

接触点数 = 0           ：脚悬空，仿真开始后机器人会掉下来
接触点数 > 0 且穿透很浅 ：脚刚好贴地，是合适的初始高度
穿透很深               ：脚陷在地面下，仿真开始后会被弹起
"""
import os
from pathlib import Path

import mujoco
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]

START, STOP, STEP = 0.5, 1.5, 0.02  # 扫描范围和步长 (m)

os.chdir(ROOT_DIR)  # MuJoCo 无法打开含中文的绝对路径
model = mujoco.MjModel.from_xml_path("t800.xml")
data = mujoco.MjData(model)


def ground_contacts():
    """返回 (接触点数, 最大穿透深度 mm)。地面是挂在 world (body 0) 上的 geom。"""
    count, depth = 0, 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
        if (b1 == 0) != (b2 == 0):          # 一边是地面，另一边是机器人
            count += 1
            depth = max(depth, -c.dist * 1000)  # dist < 0 表示穿透
    return count, depth


print(f"{'高度 z (m)':>10}  {'接触点数':>8}  {'最大穿透 (mm)':>14}")
best = None
for z in np.arange(START, STOP + 1e-9, STEP):
    mujoco.mj_resetData(model, data)  # 关节回到初始角，只改根部高度
    data.qpos[2] = z
    mujoco.mj_forward(model, data)    # 只算接触，不推进仿真
    count, depth = ground_contacts()
    print(f"{z:>10.2f}  {count:>8d}  {depth:>14.1f}")
    if count > 0:
        best = z  # 从低往高扫，最后一个还有接触的高度就是"刚好贴地"

print()
if best is None:
    print("整个范围内都没有接触，脚一直悬空")
else:
    print(f"推荐初始高度：约 {best:.2f} m（还有接触的最高高度，脚刚好贴地）")
