"""T800 站立控制：低增益关节 PD + 基于重心（COM）反馈的踝策略平衡，并在 viewer 中仿真显示。

与 12_t800_pd_hold.py 对比：
- 12 靠很大的关节 kp（1500~2000）把关节"锁死"硬撑，真机上不现实；
- 本文件关节 kp 降到 100~250 的量级，站稳靠外层的平衡反馈：
  重心相对双脚中心偏了多少、偏得多快，就在踝关节上补多少力矩，把重心拉回来。
"""
import os
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

# 工作区根目录（本文件位于 .vscode/code/）
ROOT_DIR = Path(__file__).resolve().parents[2]

# 底层关节 PD 增益：(kp, kd)，按关节名关键字匹配第一个命中的
GAINS = [
    ("HIP", (200.0, 5.0)),
    ("KNEE", (250.0, 6.0)),
    ("ANKLE", (100.0, 3.0)),
    ("TORSO", (200.0, 5.0)),
    ("SHOULDER", (50.0, 2.0)),
    ("ELBOW", (30.0, 1.0)),
    ("HEAD", (10.0, 0.5)),
]

# 踝策略平衡增益：重心偏移 (m) -> 踝力矩 (N·m)，重心速度 (m/s) -> 踝力矩
K_COM = 2000.0
D_COM = 200.0

# 推力测试：在 PUSH_TIME 时刻给机身施加 PUSH_FORCE 牛的水平力，持续 PUSH_DURATION 秒
# 实测 100 N × 0.2 s 能恢复，200 N × 0.2 s 会超出踝策略能力而摔倒；设为 (0, 0) 关闭
PUSH_FORCE = (100.0, 0.0)  # (x 向前, y 向左)
PUSH_TIME = 3.0
PUSH_DURATION = 0.2

PRINT_INTERVAL = 0.5  # 打印间隔（秒，仿真时间）


def build_controller(model):
    """为每个执行器找到对应关节的 qpos / qvel 下标，并确定 kp、kd，以及踝关节执行器的编号。"""
    names = [model.joint(model.actuator_trnid[i, 0]).name for i in range(model.nu)]
    qadr = np.array([model.joint(n).qposadr[0] for n in names])
    dadr = np.array([model.joint(n).dofadr[0] for n in names])
    kp = np.array([next(g for key, g in GAINS if key in n)[0] for n in names])
    kd = np.array([next(g for key, g in GAINS if key in n)[1] for n in names])
    ankle_pitch = [i for i, n in enumerate(names) if "ANKLE_PITCH" in n]
    ankle_roll = [i for i, n in enumerate(names) if "ANKLE_ROLL" in n]
    return qadr, dadr, kp, kd, ankle_pitch, ankle_roll


def com_offset(model, data, foot_ids):
    """整机重心相对双脚中心的水平偏移 [x, y]（世界坐标系，默认机器人朝 +x）。"""
    feet_center = data.xpos[foot_ids, :2].mean(axis=0)
    return data.subtree_com[model.body("LINK_BASE").id, :2] - feet_center


def main():
    # MuJoCo 在 Windows 上无法打开含中文的绝对路径，
    # 所以先切换到根目录，再用纯英文的相对路径加载
    os.chdir(ROOT_DIR)
    model = mujoco.MjModel.from_xml_path("t800.xml")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    qadr, dadr, kp, kd, ankle_pitch, ankle_roll = build_controller(model)
    tau_limit = model.actuator_ctrlrange
    base_id = model.body("LINK_BASE").id
    foot_ids = [model.body("LINK_FOOT_L").id, model.body("LINK_FOOT_R").id]

    q_target = data.qpos[qadr].copy()                   # 关节目标 = 初始角度
    com_target = com_offset(model, data, foot_ids)      # 重心目标 = 初始站姿下的偏移
    com_prev = com_target.copy()
    dt = model.opt.timestep
    steps_per_print = max(round(PRINT_INTERVAL / dt), 1)

    print(f"推力测试: {PUSH_FORCE} N，t = {PUSH_TIME} ~ {PUSH_TIME + PUSH_DURATION} s")
    print(f"{'t (s)':>6}  {'根部高度 z (m)':>14}  {'重心偏移 x/y (cm)':>18}  {'最大关节误差 (deg)':>18}")

    # 被动 viewer：仿真循环由我们控制，关闭窗口即退出
    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        while viewer.is_running():
            step_start = time.time()

            # 1) 底层关节 PD：低增益跟踪初始角度
            q = data.qpos[qadr]
            dq = data.qvel[dadr]
            tau = kp * (q_target - q) - kd * dq

            # 2) 踝策略平衡：重心偏了多少、偏得多快，就在踝上补力矩把重心推回去
            #    （仿真里直接读重心；真机上需要用 IMU + 运动学估计）
            com = com_offset(model, data, foot_ids)
            com_err = com - com_target
            com_vel = (com - com_prev) / dt
            com_prev = com
            # 踝 pitch 绕 +y 轴：正力矩把重心推向 -x，所以重心前偏（x>0）就补正力矩
            tau[ankle_pitch] += K_COM * com_err[0] + D_COM * com_vel[0]
            # 踝 roll 绕 +x 轴：正力矩把重心推向 +y，所以重心左偏（y>0）就补负力矩
            tau[ankle_roll] -= K_COM * com_err[1] + D_COM * com_vel[1]

            data.ctrl[:] = np.clip(tau, tau_limit[:, 0], tau_limit[:, 1])

            # 推力测试：在指定时间段给机身施加水平外力
            pushing = PUSH_TIME <= data.time < PUSH_TIME + PUSH_DURATION
            data.xfrc_applied[base_id, :2] = PUSH_FORCE if pushing else (0.0, 0.0)

            mujoco.mj_step(model, data)
            step += 1
            viewer.sync()

            if step % steps_per_print == 0:
                err = np.degrees(np.abs(q_target - data.qpos[qadr]).max())
                cx, cy = (com_offset(model, data, foot_ids) - com_target) * 100
                print(f"{data.time:>6.2f}  {data.qpos[2]:>14.4f}  {cx:>8.2f} / {cy:>7.2f}  {err:>18.3f}")

            # 按 timestep 休眠，让仿真接近实时速度
            elapsed = time.time() - step_start
            if elapsed < dt:
                time.sleep(dt - elapsed)


if __name__ == "__main__":
    main()
