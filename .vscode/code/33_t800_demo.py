"""生成 README 用的演示 GIF：机器人走路、被推、负载失控、行为克隆摔倒。

四段（命令行参数选择，默认全生成）：
  walk     教师（官方策略）和 DAgger 学生并排行走，0.5 m/s
  push     向左推 400 N：推的瞬间摆动腿在左侧时恢复，支撑腿在左侧时摔倒（0.5 倍慢放）
  payload  负载 +42% 和 +44%：速度越走越快，最后失控摔倒
  bc       纯行为克隆学生和 DAgger 学生：复合误差

用法：
  python 33_t800_demo.py
  python 33_t800_demo.py push payload
  python 33_t800_demo.py --preview          # 另存每段的 4 张截图，用来检查画面

结果：results/demo/<名字>.gif。所有场景用的参数都和评测实验一致（种子 0），
每段右上角的"摔倒 / 未摔倒"是仿真的真实结果，不是写死的。
"""
import argparse
import importlib.util
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CODE_DIR = Path(__file__).resolve().parent
OUT_DIR = CODE_DIR / "results" / "demo"

# ---------- 画面设置 ----------
W, H = 320, 240                 # 每个画面的大小
GAP = 4                         # 并排画面之间的间隔
RENDER_EVERY = 5                # 每 5 个策略步（50 ms）渲染一帧，即 20 fps
FRAME_MS = 50                   # GIF 每帧 50 ms = 实时速度
HOLD_MS = 1500                  # 结尾定格时间
TAIL_S = 0.8                    # 摔倒后继续仿真多久，让机器人倒到地上
RED, GREEN = (227, 73, 72), (27, 175, 122)
FONT_FILES = ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/simsun.ttc"]

# ---------- 场景设置 ----------
PUSH_FORCE = 400.0              # 向左 (+y) 的推力 (N)
PUSH_SWING_T = 7.7168           # 步态相位 72%：右脚支撑，左腿在摆动。±4 ms 抖动下都恢复（相位 74% 对 2 ms 敏感，不用）
PUSH_STANCE_T = 6.902           # push_phase.csv 里步态相位 30%：左脚支撑
PAYLOADS = (0.42, 0.44)         # 刚好不摔 vs 一定摔
BC_MODEL, DAGGER_MODEL = "bc_mlp.pt", "bc_mlp_r4.pt"


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, CODE_DIR / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


t25 = load("r25", "25_t800_robustness.py")   # 注意：导入后当前目录变成了 MNN-CODE，路径一律用绝对路径
t27 = load("t27", "27_t800_bc_train.py")


def font(size):
    for f in FONT_FILES:
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


FONT_M, FONT_S = font(15), font(13)


def load_student(name):
    return t27.load_student(str(CODE_DIR / "models" / name))[1]


def render_clip(policy, *, window, azimuth=135, vx=0.5, push=None, t_push=None, payload=0.0, seed=0):
    """跑一次仿真，在 window（仿真时间 s）内录画面。
    返回 (原始画面列表, 是否摔倒, run_trial 的返回值)。画面里每帧附带 (时刻, 平滑速度, 是否正在被推)。"""
    m, d = t25.model, t25.data
    t25.set_dynamics(1.0, payload)
    renderer = mujoco.Renderer(m, H, W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = t25.base_id
    cam.distance, cam.azimuth, cam.elevation = 3.2, azimuth, -12
    raw, vx_hist, calls = [], deque(maxlen=50), [0]    # 速度取最近 0.5 s 的平均

    def snap():
        renderer.update_scene(d, camera=cam)
        pushing = push is not None and t_push <= d.time < t_push + t25.PUSH_DURATION
        return renderer.render().copy(), dict(t=d.time, vx=float(np.mean(vx_hist)) if vx_hist else 0.0,
                                              pushing=pushing)

    def wrapped(obs):
        calls[0] += 1
        vx_hist.append(d.qvel[0])
        if calls[0] % RENDER_EVERY == 0 and window[0] <= d.time <= window[1]:
            raw.append(snap())
        return policy(obs)

    kw = dict(push=push, t_push=t_push) if push is not None else {}
    res = t25.run_trial(vx, 0.0, seed=seed, policy=wrapped, **kw)
    fell = bool(res[0])
    if fell and res[1] <= window[1]:           # run_trial 一判定摔倒就返回，这里让机器人倒到地上再录一小段
        d.ctrl[:] = 0
        d.xfrc_applied[:] = 0
        for k in range(round(TAIL_S / t25.DT)):
            mujoco.mj_step(m, d)
            if k % 25 == 0:
                raw.append(snap())
    renderer.close()
    t25.set_dynamics()                          # 还原摩擦和负载
    return raw, fell, res


def decorate(raw, title, fell, *, show_t=True, show_vx=False, slow=False, push_label=""):
    """给画面加标题、时间、速度、推力提示，最后 1 秒挂上结果。"""
    out = []
    badge_from = max(0, len(raw) - round(1000 / FRAME_MS))
    for i, (img, meta) in enumerate(raw):
        im = Image.fromarray(img)
        dr = ImageDraw.Draw(im)
        dr.rectangle((0, 0, W, 24), fill=(18, 20, 28))
        dr.text((6, 3), title, font=FONT_M, fill=(240, 239, 236))
        dr.rectangle((0, H - 22, W, H), fill=(18, 20, 28))
        left = ("0.5× 慢放" if slow else f"t = {meta['t']:.1f} s") if (slow or show_t) else ""
        if left:
            dr.text((6, H - 19), left, font=FONT_S, fill=(195, 194, 183))
        if show_vx:
            dr.text((W - 118, H - 19), f"vx = {meta['vx']:.2f} m/s", font=FONT_S, fill=(240, 239, 236))
        if meta["pushing"]:
            dr.rectangle((0, H - 48, W, H - 24), fill=RED)
            dr.text((6, H - 45), push_label, font=FONT_M, fill=(255, 255, 255))
        if i >= badge_from:
            dr.rectangle((W - 84, 30, W - 6, 54), fill=RED if fell else GREEN)
            dr.text((W - 76, 33), "摔倒" if fell else "未摔倒", font=FONT_M, fill=(255, 255, 255))
        out.append(im)
    return out


def combine(panels, name, ms, preview=False):
    """把几个画面左右并排，短的定格在最后一帧，保存为 GIF。"""
    n = max(len(p) for p in panels)
    panels = [p + [p[-1]] * (n - len(p)) for p in panels]
    frames = []
    for i in range(n):
        canvas = Image.new("RGB", (W * len(panels) + GAP * (len(panels) - 1), H), (255, 255, 255))
        for k, p in enumerate(panels):
            canvas.paste(p[i], (k * (W + GAP), 0))
        frames.append(canvas)
    frames += [frames[-1]] * round(HOLD_MS / ms)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pal = [f.quantize(colors=128, method=Image.Quantize.FASTOCTREE) for f in frames]
    path = OUT_DIR / f"{name}.gif"
    pal[0].save(path, save_all=True, append_images=pal[1:], duration=ms, loop=0, optimize=True, disposal=2)
    print(f"  -> results/demo/{name}.gif  {len(frames)} 帧  {path.stat().st_size / 1e6:.1f} MB")
    if preview:
        idx = [0, len(frames) // 3, 2 * len(frames) // 3, len(frames) - 1]
        sheet = Image.new("RGB", (frames[0].width, H * 4))
        for k, j in enumerate(idx):
            sheet.paste(frames[j], (0, k * H))
        sheet.save(OUT_DIR / f"_preview_{name}.png")


def clip_walk(preview):
    teacher, f1, _ = render_clip(t25.infer, window=(3, 9))
    student, f2, _ = render_clip(load_student(DAGGER_MODEL), window=(3, 9))
    print(f"  教师 摔倒={f1}，DAgger 学生 摔倒={f2}")
    combine([decorate(teacher, "教师：官方策略", f1, show_vx=True),
             decorate(student, "学生：DAgger", f2, show_vx=True)], "walk", FRAME_MS, preview)


def clip_push(preview):
    panels = []
    for t_push in (PUSH_SWING_T, PUSH_STANCE_T):
        raw, fell, res = render_clip(t25.infer, window=(t_push - 1.0, t_push + 3.0), azimuth=160,
                                     push=(0.0, PUSH_FORCE), t_push=t_push)
        print(f"  t_push={t_push}  推的瞬间 {res[5]}  摔倒={fell}")
        panels.append(decorate(raw, f"向左推 {PUSH_FORCE:.0f} N，{res[5]}支撑", fell, slow=True,
                               push_label=f"推力 {PUSH_FORCE:.0f} N  向左"))
    combine(panels, "push", FRAME_MS * 2, preview)


def clip_payload(preview):
    panels = []
    for p in PAYLOADS:
        raw, fell, _ = render_clip(t25.infer, window=(0, 9), payload=p)
        print(f"  负载 +{p:.0%}  摔倒={fell}")
        panels.append(decorate(raw, f"负载 +{p:.0%}（{p * t25.ROBOT_MASS:.1f} kg）", fell, show_vx=True))
    combine(panels, "payload", FRAME_MS, preview)


def clip_bc(preview):
    panels = []
    for model_name, title in ((BC_MODEL, "行为克隆学生"), (DAGGER_MODEL, "DAgger 学生")):
        raw, fell, _ = render_clip(load_student(model_name), window=(0, 9))
        print(f"  {title}  摔倒={fell}")
        panels.append(decorate(raw, title, fell))
    combine(panels, "bc", FRAME_MS, preview)


CLIPS = {"walk": clip_walk, "push": clip_push, "payload": clip_payload, "bc": clip_bc}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clips", nargs="*", help="要生成的片段：" + " / ".join(CLIPS) + "，默认全部")
    parser.add_argument("--preview", action="store_true", help="另存每段的 4 张截图用来检查")
    args = parser.parse_args()
    for name in args.clips or list(CLIPS):
        if name not in CLIPS:
            parser.error(f"未知片段 {name}，可选：{list(CLIPS)}")
        print(f"\n===== {name} =====")
        CLIPS[name](args.preview)
