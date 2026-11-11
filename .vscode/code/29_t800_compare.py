"""第 4 轮 DAgger 学生 vs 老师在完全相同条件下的对比（阶段 2 交付物）。

三个实验，命令行参数选择：
  speed     速度跟踪：vx = 0.3~0.6 每 0.1 一档，每档 10 个种子
  push      行走中抗推：前 / 后 / 左 / 右四个方向，推力从 50 N 起每档 25 N（连续 2 档全摔就停）
  latency   动作延迟：0~60 ms 每 10 ms 一档，每档 10 个种子

老师和学生跑同一批条件和种子，只换 run_trial(policy=...) 的取值，
其余（观测、PD、摔倒判断、航向闭环）完全一致。

用法：
  python 29_t800_compare.py speed
  python 29_t800_compare.py push
  python 29_t800_compare.py latency
  python 29_t800_compare.py speed --model models/bc_mlp.pt    # 换别的权重对比

结果：results/compare_<实验>.csv，屏幕上也直接打印表格。
"""
import argparse
import csv
import importlib.util
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
RESULT_DIR = CODE_DIR / "results"

# ---------- 对比设置 ----------
SEEDS = range(10)                 # 每档 10 个种子（25 的报告里是 5 个，这里加倍让临界点更稳）
SPEED_CMDS = (0.3, 0.4, 0.5, 0.6)
PUSH_DIRS = {"向前 (+x)": (1, 0), "向后 (-x)": (-1, 0), "向左 (+y)": (0, 1), "向右 (-y)": (0, -1)}
PUSH_FORCES = np.arange(50, 801, 25)
DELAYS_MS = range(0, 61, 10)
DEFAULT_MODEL = "models/bc_mlp_r4.pt"
PASS_RATE = 0.8

spec = importlib.util.spec_from_file_location("r25", CODE_DIR / "25_t800_robustness.py")
t25 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t25)

spec27 = importlib.util.spec_from_file_location("t27", CODE_DIR / "27_t800_bc_train.py")
t27 = importlib.util.module_from_spec(spec27)
spec27.loader.exec_module(t27)


def trial_stats(policy, **kw):
    """跑一批种子，返回 (摔倒率, 实际 vx)；全摔时平均速度为 nan。"""
    res = [t25.run_trial(seed=s, policy=policy, **kw) for s in SEEDS]
    return float(t25.fall_rate(res)), float(t25.mean_speed(res, 2))


def fmt(v, nd=3):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def cmd_speed(student):
    rows = []
    print(f"\n速度跟踪（每档 {len(SEEDS)} 个种子，15 s）")
    print(f"{'指令vx':>7} | {'老师 实际vx':>11} {'误差':>6} {'摔倒率':>7} | {'学生 实际vx':>11} {'误差':>6} {'摔倒率':>7}")
    for vx in SPEED_CMDS:
        out = {}
        for name, pol in (("老师", None), ("学生", student)):
            fr, avx = trial_stats(pol, vx=vx, vy=0.0)
            out[name] = (avx, fr)
            rows.append([vx, name, round(avx, 3) if not np.isnan(avx) else "", 
                         round(abs(avx - vx), 3) if not np.isnan(avx) else "", round(fr, 3)])
        t_avx, t_fr = out["老师"]
        s_avx, s_fr = out["学生"]
        print(f"{vx:>7.1f} | {fmt(t_avx):>11} {fmt(abs(t_avx - vx) if not np.isnan(t_avx) else np.nan):>6} "
              f"{t_fr:>6.0%} | {fmt(s_avx):>11} {fmt(abs(s_avx - vx) if not np.isnan(s_avx) else np.nan):>6} "
              f"{s_fr:>6.0%}")
    t27.write_csv(RESULT_DIR / "compare_speed.csv", ["指令vx", "策略", "实际vx", "跟踪误差", "摔倒率"], rows)


def cmd_push(student):
    rows = []
    print(f"\n行走中抗推（0.5 m/s，推 0.2 s，每档 {len(SEEDS)} 个种子）")
    for name, (ux, uy) in PUSH_DIRS.items():
        print(f"  {name}：")
        print(f"    {'推力(N)':>7} {'老师摔倒率':>10} {'学生摔倒率':>10}")
        zero = {"老师": 0, "学生": 0}
        for f in PUSH_FORCES:
            line = {}
            for who, pol in (("老师", None), ("学生", student)):
                if zero[who] >= 2:      # 连续 2 档全摔就不再往上加了
                    continue
                fr, _ = trial_stats(pol, push=(f * ux, f * uy))
                line[who] = fr
                rows.append([name, int(f), who, round(fr, 3)])
                zero[who] = zero[who] + 1 if fr == 1.0 else 0
            if line:
                print(f"    {f:>7} {line.get('老师', float('nan')):>10.0%} {line.get('学生', float('nan')):>10.0%}")
            if all(zero[w] >= 2 for w in zero):
                break
    t27.write_csv(RESULT_DIR / "compare_push.csv", ["方向", "推力(N)", "策略", "摔倒率"], rows)


def cmd_latency(student, delays, append=False):
    rows = []
    print(f"\n动作延迟（0.5 m/s，每档 {len(SEEDS)} 个种子，15 s）")
    print(f"{'延迟(ms)':>8} | {'老师 摔倒率':>11} {'实际vx':>7} | {'学生 摔倒率':>11} {'实际vx':>7}")
    for d in delays:
        out = {}
        for who, pol in (("老师", None), ("学生", student)):
            fr, avx = trial_stats(pol, vx=0.5, vy=0.0, delay_ms=d)
            out[who] = (fr, avx)
            rows.append([d, who, round(fr, 3), round(avx, 3) if not np.isnan(avx) else ""])
        print(f"{d:>8} | {out['老师'][0]:>10.0%} {fmt(out['老师'][1]):>7} | "
              f"{out['学生'][0]:>10.0%} {fmt(out['学生'][1]):>7}")
    out_path = RESULT_DIR / "compare_latency.csv"
    if append and out_path.exists():  # 补测单档：把新结果并进已有 CSV，按延迟排序
        with open(out_path, encoding="utf-8-sig") as f:
            rows = list(csv.reader(f))[1:] + rows
        order = {"老师": 0, "学生": 1}
        rows.sort(key=lambda r: (int(r[0]), order.get(r[1], 9)))
    t27.write_csv(out_path, ["延迟ms", "策略", "摔倒率", "实际vx"], rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", choices=["speed", "push", "latency"])
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"学生权重，默认 {DEFAULT_MODEL}")
    parser.add_argument("--delays", default="", help="只测指定的延迟档位，逗号分隔，例如 45（配合 --append 补测）")
    parser.add_argument("--append", action="store_true", help="把结果并进已有的 compare_latency.csv")
    args = parser.parse_args()

    net, student, ckpt = t27.load_student(args.model)
    print(f"学生 {args.model}：训练 {ckpt['epoch']} 轮，验证 MSE {ckpt['val_loss']:.5f}；"
          f"每档 {len(SEEDS)} 个种子")
    if args.experiment == "latency":
        delays = [int(d) for d in args.delays.split(",")] if args.delays else list(DELAYS_MS)
        cmd_latency(student, delays, args.append)
    elif args.experiment == "speed":
        cmd_speed(student)
    else:
        cmd_push(student)


if __name__ == "__main__":
    main()
