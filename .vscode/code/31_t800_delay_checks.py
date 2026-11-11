"""阶段 3 的补充检查（跑完弹一次窗）。三件事按顺序做：

 A. 补测 45 ms：对照 r2 / 对照 r3 / 主方案 r1，各 10 个种子，追加到对应的 CSV；
 B. d = 0 的速度跟踪 + 抗推：对照 r2、主方案 r1（和阶段 2 学生的数字并列比较）；
 C. 训练种子波动：主方案 r1、对照 r1 用训练种子 1、2 重新训练（采集数据不变），
    测 0~60 ms + 45 ms 的延迟曲线。

用法：python 31_t800_delay_checks.py
"""
import csv
import importlib.util
import time
import traceback
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent
RES = CODE / "results"

spec = importlib.util.spec_from_file_location("d30", CODE / "30_t800_delay_student.py")
d30 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d30)          # 会加载 25（MuJoCo + MNN）和 27
t27 = d30.t27


def read_rows(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.reader(f))


# ---------- A. 补测 45 ms ----------
def append_45(csv_name, model, tag):
    """测一个模型在 45 ms 下的摔倒率，追加进指定 CSV（按延迟、策略排序）。"""
    net = t27.load_student(model)[0]
    fr, avx = d30.trial(d30.student_fn(net, 45), 45)
    path = RES / csv_name
    rows = read_rows(path)
    header, body = rows[0], [r for r in rows[1:] if not (int(r[0]) == 45 and r[1] == tag)]
    body.append([45, tag, round(fr, 3), round(avx, 3) if not np.isnan(avx) else "", "是"])
    order = {"学生_r1": 0, "学生_r2": 1, "学生_r3": 2}
    body.sort(key=lambda r: (int(r[0]), order.get(r[1], 9)))
    t27.write_csv(path, header, body)
    print(f"  {tag} 45 ms：摔倒率 {fr:.0%}，实际 vx {avx:.3f} → 写入 {csv_name}", flush=True)
    return fr


# ---------- B. d = 0 的速度与抗推 ----------
def speed_push(model, label):
    net = t27.load_student(model)[0]
    print(f"  {label}：d=0 速度跟踪", flush=True)
    speed = [[r[0], label] + list(r[2:]) for r in d30.eval_speed(net)]
    print(f"  {label}：d=0 抗推", flush=True)
    push = [[r[0], r[1], label, r[3]] for r in d30.eval_push(net)]
    return speed, push


# ---------- C. 训练种子波动 ----------
def seed_run(shift, train_seed):
    """只用第 1 轮的数据、换训练种子重新训练，返回延迟行。"""
    suffix = "" if shift else "_noshift"
    paths = [CODE / p for p in d30.BASE_DATA] + [d30.DATA_DIR / f"delay_r1{suffix}.npz"]
    data = d30.load_all(paths)
    label = f"{'主' if shift else '对照'}r1_s{train_seed}"
    t0 = time.time()
    net, best, dt = d30.train_model(*data, d30.EPOCHS, label, seed=train_seed)
    print(f"  {label} 训练完成（{dt:.0f} s，验证 MSE {best:.5f}）", flush=True)
    return d30.eval_latency(net, label, [0, 10, 20, 30, 40, 45, 50, 60])


def run():
    t_start = time.time()

    print("=== A. 补测 45 ms ===", flush=True)
    a = {
        "对照 r2": append_45("delay_latency.csv", "models/delay_mlp_r2_noshift.pt", "学生_r2"),
        "对照 r3": append_45("delay_latency.csv", "models/delay_mlp_r3_noshift.pt", "学生_r3"),
        "主方案 r1": append_45("delay_latency_main.csv", "models/delay_mlp_r1.pt", "学生_r1"),
    }

    print("=== B. d=0 速度跟踪与抗推 ===", flush=True)
    speed_rows, push_rows = [], []
    for model, label in (("models/delay_mlp_r2_noshift.pt", "对照r2"), ("models/delay_mlp_r1.pt", "主方案r1")):
        s, p = speed_push(model, label)
        speed_rows += s
        push_rows += p
    t27.write_csv(RES / "delay_speed_extra.csv", ["指令vx", "模型", "实际vx", "跟踪误差", "摔倒率"], speed_rows)
    t27.write_csv(RES / "delay_push_extra.csv", ["方向", "推力(N)", "模型", "摔倒率"], push_rows)

    print("=== C. 训练种子波动（只训第 1 轮）===", flush=True)
    seed_rows = []
    for shift in (True, False):
        for s in (1, 2):
            seed_rows += seed_run(shift, s)
    t27.write_csv(RES / "delay_seed_variance.csv", ["延迟ms", "模型", "摔倒率", "实际vx", "超出训练范围(>60ms)"],
                  seed_rows)

    print(f"全部完成，用时 {(time.time() - t_start) / 60:.0f} 分钟", flush=True)
    return (f"补充检查完成（{a['对照 r2']:.0%}/{a['对照 r3']:.0%}/{a['主方案 r1']:.0%} @45ms；"
            f"种子波动见 delay_seed_variance.csv）")


if __name__ == "__main__":
    try:
        summary = run()
    except Exception:
        d30.notify("T800 任务出错", traceback.format_exc().strip().splitlines()[-1], error=True)
        raise
    else:
        d30.notify("T800 任务完成", summary)
