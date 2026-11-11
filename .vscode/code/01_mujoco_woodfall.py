"""MuJoCo 木块自由落体仿真：木块从 1 米高度落下，每隔 200 毫秒打印一次高度。"""
import mujoco

# 场景：地面 + 一个边长 0.1 m 的木块（密度约 600 kg/m^3）
XML = """
<mujoco model="wood_fall">
  <option timestep="0.001" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.8 0.8 0.8 1"/>
    <body name="wood" pos="0 0 1">
      <freejoint/>
      <geom name="wood_geom" type="box" size="0.05 0.05 0.05"
            density="600" rgba="0.6 0.4 0.2 1"/>
    </body>
  </worldbody>
</mujoco>
"""

PRINT_INTERVAL = 0.2  # 打印间隔，单位秒（200 毫秒）
SIM_DURATION = 1.0    # 仿真总时长，单位秒


def main():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wood")

    # 每多少个仿真步打印一次
    steps_per_print = round(PRINT_INTERVAL / model.opt.timestep)
    total_steps = round(SIM_DURATION / model.opt.timestep)

    mujoco.mj_forward(model, data)
    print(f"t = {data.time:.1f} s, 高度 = {data.xpos[body_id][2]:.4f} m")

    for step in range(1, total_steps + 1):
        mujoco.mj_step(model, data)
        if step % steps_per_print == 0:
            # xpos 是木块中心的世界坐标，z 分量即高度
            print(f"t = {data.time:.1f} s, 高度 = {data.xpos[body_id][2]:.4f} m")


if __name__ == "__main__":
    main()
