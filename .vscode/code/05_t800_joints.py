"""加载 T800 的 MuJoCo 模型，打印所有关节的名称、类型和范围。"""
import os
from pathlib import Path

import mujoco

# 工作区根目录（本文件位于 .vscode/code/）
ROOT_DIR = Path(__file__).resolve().parents[2]

# 关节类型编号 -> 名称
JOINT_TYPES = {
    mujoco.mjtJoint.mjJNT_FREE: "free",
    mujoco.mjtJoint.mjJNT_BALL: "ball",
    mujoco.mjtJoint.mjJNT_SLIDE: "slide",
    mujoco.mjtJoint.mjJNT_HINGE: "hinge",
}


def main():
    # MuJoCo 在 Windows 上无法打开含中文的绝对路径，
    # 所以先切换到根目录，再用纯英文的相对路径加载
    os.chdir(ROOT_DIR)
    model = mujoco.MjModel.from_xml_path("t800.xml")

    print(f"关节总数: {model.njnt}")
    # 中文字符显示宽度为 2，所以表头的填充宽度比数据行少 2
    print(f"{'id':>3}  {'名称':<28}{'类型':<8}{'范围 (rad)'}")
    for i in range(model.njnt):
        joint = model.joint(i)
        jnt_type = JOINT_TYPES[joint.type[0]]
        name = joint.name or "(未命名)"
        # limited 为 0 表示该关节没有限位（如 freejoint）
        if model.jnt_limited[i]:
            low, high = joint.range
            range_str = f"[{low:.3f}, {high:.3f}]"
        else:
            range_str = "无限位"
        # "(未命名)" 含 3 个中文字符，多占 3 格显示宽度
        width = 30 - (3 if not joint.name else 0)
        print(f"{i:>3}  {name:<{width}}{jnt_type:<10}{range_str}")


if __name__ == "__main__":
    main()
