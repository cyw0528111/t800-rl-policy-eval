"""画阶段 3 的延迟曲线（跑完弹窗通知）。

左图：主方案 r1/r2/r3、对照组 r1/r2/r3、老师 的延迟曲线；
右图：训练种子波动（主 r1 与对照 r1 各 3 个训练种子）。
缺失的档位（例如对照 r1 没测 45 ms）留空，不做填补。

用法：python 32_t800_delay_plot.py
输出：results/delay_curves.png
"""
import csv
import ctypes
import traceback
import winsound
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path(__file__).resolve().parent / "results"
GRID = [0, 10, 20, 30, 40, 45, 50, 60, 70, 80]   # 横轴档位（45 ms 是补测点）


def notify(title, msg, error=False):
    winsound.MessageBeep(winsound.MB_ICONHAND if error else winsound.MB_ICONASTERISK)
    ctypes.windll.user32.MessageBoxW(0, msg, title, (0x10 if error else 0x40) | 0x40000 | 0x10000)


def series(path, key_col, key):
    """从 CSV 里取一条曲线：{延迟ms: 摔倒率%}，没有的档位就不放键（画图时留空）。"""
    out = {}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row[key_col] == key:
                out[int(row["延迟ms"])] = float(row["摔倒率"]) * 100
    return out


def draw(ax, data, color, label):
    if not data:
        print(f"  警告：{label} 没有数据，曲线为空", flush=True)
    y = [data.get(x, float("nan")) for x in GRID]   # 缺的档位用 nan，线会断开，不插值
    ax.plot(GRID, y, "-o", color=color, linewidth=2, markersize=6, label=label)


def make_fig():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    # 左图：主方案 / 对照组 / 老师
    draw(ax1, series(RES / "compare_latency.csv", "策略", "老师"), "#0b0b0b", "老师")
    for color, tag, name in (("#2a78d6", "学生_r1", "主方案 r1"), ("#1b5fa8", "学生_r2", "主方案 r2"),
                             ("#6ba3e8", "学生_r3", "主方案 r3")):
        draw(ax1, series(RES / "delay_latency_main.csv", "策略", tag), color, name)
    for color, tag, name in (("#eb6834", "学生_r1", "对照组 r1"), ("#b8491d", "学生_r2", "对照组 r2"),
                             ("#f2a17b", "学生_r3", "对照组 r3")):
        draw(ax1, series(RES / "delay_latency.csv", "策略", tag), color, name)
    ax1.set_title("延迟曲线：主方案（蓝，标签前移）/ 对照组（橙，不前移）/ 老师（黑）")
    ax1.legend(frameon=False, ncol=2, fontsize=9)

    # 右图：训练种子波动（只训第 1 轮）
    seed = RES / "delay_seed_variance.csv"
    draw(ax2, series(RES / "delay_latency_main.csv", "策略", "学生_r1"), "#2a78d6", "主 r1 种子 0")
    draw(ax2, series(seed, "模型", "学生_主r1_s1"), "#4c8fe0", "主 r1 种子 1")
    draw(ax2, series(seed, "模型", "学生_主r1_s2"), "#8fbdf0", "主 r1 种子 2")
    draw(ax2, series(RES / "delay_latency.csv", "策略", "学生_r1"), "#eb6834", "对照 r1 种子 0")
    draw(ax2, series(seed, "模型", "学生_对照r1_s1"), "#ef8a5c", "对照 r1 种子 1")
    draw(ax2, series(seed, "模型", "学生_对照r1_s2"), "#f5bd9d", "对照 r1 种子 2")
    ax2.set_title("训练种子波动（同一份采集数据，只换训练种子）")
    ax2.legend(frameon=False, ncol=2, fontsize=9)

    for ax in (ax1, ax2):
        ax.set(xlabel="动作延迟 (ms)", ylabel="摔倒率 (%)", ylim=(-5, 105))
        ax.set_xticks(GRID)
        ax.grid(alpha=0.3, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = RES / "delay_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def run():
    out = make_fig()
    print(f"图已保存：{out}", flush=True)
    return f"延迟曲线图已生成：results/{out.name}"


if __name__ == "__main__":
    try:
        summary = run()
    except Exception:
        notify("T800 任务出错", traceback.format_exc().strip().splitlines()[-1], error=True)
        raise
    else:
        notify("T800 任务完成", summary)
