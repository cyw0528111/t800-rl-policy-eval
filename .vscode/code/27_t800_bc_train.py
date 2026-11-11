"""行为克隆：用采集到的老师数据训练一个 MLP 学生策略（阶段 1）。

输入 1083 维（15 帧历史 × 72 + 指令 3），输出 22 维动作，MSE 损失，Adam 优化器。
数据集按"回合"划分训练 / 验证：不能按帧随机划分，因为相邻帧几乎一样，会把验证集泄漏进训练集。

数据里存的是 72 维单帧，训练时按下面的规则拼成 15 帧历史（和 25 的 run_trial、26 的采集完全一致，
已经用 _check_collect.py 验证过，重建观测喂给老师能复现记录的动作，偏差为 0）：
    第 k 步的历史 = frames[clip(k-14+t, 0, k)]，t = 0..14
也就是前 14 步不够时，用第一帧补在前面。

用法：
  python 27_t800_bc_train.py --epochs 2 --limit 8000   # 小规模试跑，确认流程
  python 27_t800_bc_train.py                           # 默认 60 轮
  python 27_t800_bc_train.py --eval models/bc_mlp.pt   # 只用训练好的模型跑评测（阶段 1 第 3 步）

输出：
  models/bc_mlp.pt            权重 + 网络结构配置
  results/bc_train.csv        每轮的训练 / 验证损失
  results/bc_train_curve.png  损失曲线
  results/bc_eval.csv/.png    评测结果（--eval 时）
"""
import argparse
import csv
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")  # 不弹窗口，直接存图
import matplotlib.pyplot as plt

CODE_DIR = Path(__file__).resolve().parent
RESULT_DIR = CODE_DIR / "results"
MODEL_DIR = CODE_DIR / "models"

# ---------- 训练设置 ----------
DATA_FILE = "data/bc_teacher.npz"
VAL_RATIO = 0.1        # 验证集占的回合比例（按回合划分）
EXCLUDE_FALLEN = True  # 摔倒的回合不参与训练（200 回合里只有个位数）
HIDDEN = (512, 256)    # MLP 隐藏层
BATCH = 1024          # CPU 上批大一点吞吐更高（实测 1024 时一轮约 7 s）
EPOCHS = 60
LR = 1e-3
SEED = 0
MODEL_PATH = "models/bc_mlp.pt"
EVAL_CMDS = (0.3, 0.4, 0.5, 0.6)   # 评测用的速度指令
EVAL_SEEDS = range(5)              # 每个指令跑 5 次，种子 0~4，和 25 的报告一致


def load_data(path):
    """读 npz，顺便读同名的 csv 拿到每个回合是否摔倒（摔倒回合默认不参与训练）。"""
    path = Path(path)
    d = np.load(path)
    data = {k: d[k] for k in ("frames", "actions", "cmds", "episode", "step")}
    side = path.with_suffix(".csv")
    if side.exists():
        with open(side, encoding="utf-8-sig") as f:
            data["fell"] = np.array([row["是否摔倒"] == "摔倒" for row in csv.DictReader(f)])
    return data


def history_index(episode, step, history):
    """每一步对应的 15 帧历史在 frames 里的下标（同一回合内取，前 14 步用第一帧补）。"""
    n_ep = int(episode.max()) + 1
    starts = np.zeros(n_ep + 1, np.int64)               # starts[e] = 第 e 个回合第一帧的位置
    np.cumsum(np.bincount(episode, minlength=n_ep), out=starts[1:])
    offsets = np.arange(history)
    local = np.clip(step[:, None] - (history - 1) + offsets[None, :], 0, step[:, None])
    return starts[episode][:, None] + local


def make_obs(frames, cmds, idx):
    """按下标数组取 15 帧，拼成 (B, 1080) 再和指令拼成 (B, 1083)。"""
    return np.concatenate([frames[idx].reshape(len(idx), -1), cmds[idx[:, -1]]], axis=1)


def build_net(in_dim, out_dim, hidden=HIDDEN):
    layers, last = [], in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), nn.ELU()]
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


def batch_loss(net, loss_fn, frames, actions, cmds, rows, idx_all, batch=2048):
    """在给定的行上算平均损失（验证用，分块防止一次性占太多内存）。"""
    total = 0.0
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            b = rows[i:i + batch]
            x = torch.from_numpy(make_obs(frames, cmds, idx_all[b]))
            total += loss_fn(net(x), torch.from_numpy(actions[b])).item() * len(b)
    return total / len(rows)


def load_r25():
    """加载 25（评测和常量的唯一来源）；加载后当前目录会变成 MNN-CODE，所以文件路径都用绝对路径。"""
    spec = importlib.util.spec_from_file_location("r25", CODE_DIR / "25_t800_robustness.py")
    r25 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r25)
    return r25


def train(args):
    r25 = load_r25()
    H = r25.HISTORY
    d = load_data(CODE_DIR / args.data)
    frames, actions, cmds, episode, step = d["frames"], d["actions"], d["cmds"], d["episode"], d["step"]
    n_ep = int(episode.max()) + 1

    # 摔倒的回合不用来训练：摔倒前那几帧的动作已经平衡不住了，学它有副作用
    dropped_eps = 0
    if EXCLUDE_FALLEN and "fell" in d:
        fall_eps = np.flatnonzero(d["fell"][:n_ep])
        if 0 < fall_eps.size < n_ep:
            keep = ~np.isin(episode, fall_eps)
            frames, actions, cmds, step = frames[keep], actions[keep], cmds[keep], step[keep]
            episode = np.unique(episode[keep], return_inverse=True)[1].astype(np.int32)  # 重新连续编号
            dropped_eps, n_ep = int(fall_eps.size), int(episode.max()) + 1

    # 历史帧下标要在过滤之后算，保证下标指向过滤后的 frames
    idx_all = history_index(episode, step, H)

    # 按回合划分训练 / 验证，避免相邻帧泄漏
    rng = np.random.default_rng(SEED)
    n_val = max(1, round(n_ep * VAL_RATIO))
    val_eps = rng.permutation(n_ep)[:n_val]
    is_val = np.isin(episode, val_eps)
    train_rows, val_rows = np.flatnonzero(~is_val), np.flatnonzero(is_val)
    if args.limit:  # 小规模试跑：只用一部分步
        train_rows = train_rows[::max(1, len(train_rows) // args.limit)][:args.limit]
        val_rows = val_rows[::max(1, len(val_rows) // max(1, args.limit // 10))][:max(1, args.limit // 10)]

    in_dim = H * frames.shape[1] + cmds.shape[1]
    print(f"数据 {args.data}：{n_ep} 个回合 {len(frames)} 步，训练 {len(train_rows)} 步 / 验证 {len(val_rows)} 步"
          f"（验证集是 {n_val} 个整回合，剔除 {dropped_eps} 个摔倒回合），网络 {in_dim} → {HIDDEN} → {actions.shape[1]}")

    torch.manual_seed(SEED)
    net = build_net(in_dim, actions.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    (MODEL_DIR).mkdir(exist_ok=True)
    RESULT_DIR.mkdir(exist_ok=True)
    best, rows_csv, t_start = np.inf, [], time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        net.train()
        perm = rng.permutation(train_rows)
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            x = torch.from_numpy(make_obs(frames, cmds, idx_all[b]))
            loss = loss_fn(net(x), torch.from_numpy(actions[b]))
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        tr = batch_loss(net, loss_fn, frames, actions, cmds, train_rows, idx_all)
        va = batch_loss(net, loss_fn, frames, actions, cmds, val_rows, idx_all)
        improved = va < best
        if improved:  # 只保存验证损失最好的权重
            best = va
            torch.save(dict(state_dict=net.state_dict(), in_dim=in_dim, out_dim=actions.shape[1], hidden=HIDDEN,
                            history=H, epoch=epoch, val_loss=va, data=args.data), CODE_DIR / MODEL_PATH)
        rows_csv.append([epoch, round(tr, 6), round(va, 6), round(time.time() - t0, 1)])
        print(f"  第 {epoch:>3} 轮  训练 {tr:.5f}  验证 {va:.5f}  用时 {time.time() - t0:5.1f} s"
              f"{'  ← 目前最好' if improved else ''}", flush=True)

    write_csv(RESULT_DIR / "bc_train.csv", ["轮次", "训练MSE", "验证MSE", "用时(s)"], rows_csv)
    plot_curve(rows_csv)
    with open(RESULT_DIR / "bc_train_config.json", "w", encoding="utf-8") as f:
        json.dump(dict(data=args.data, episodes=n_ep, steps=int(len(frames)), train_steps=int(len(train_rows)),
                       val_steps=int(len(val_rows)), hidden=list(HIDDEN), batch=BATCH, epochs=args.epochs, lr=LR,
                       seed=SEED, best_val_mse=best), f, ensure_ascii=False, indent=2)
    print(f"完成：最好验证 MSE {best:.5f}，用时 {time.time() - t_start:.0f} s，"
          f"权重 {MODEL_PATH}，曲线 results/bc_train_curve.png")


# ---------- 保存和画图（风格和 25 一致） ----------
def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig 方便 Excel 打开
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def plot_curve(rows):
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ep = [r[0] for r in rows]
    ax.plot(ep, [r[1] for r in rows], color="#2a78d6", linewidth=2, label="训练集")
    ax.plot(ep, [r[2] for r in rows], color="#eb6834", linewidth=2, label="验证集")
    ax.set(xlabel="轮次", ylabel="MSE 损失", title="行为克隆：用 MLP 模仿老师策略", yscale="log")
    ax.grid(alpha=0.3, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "bc_train_curve.png", dpi=150)
    plt.close(fig)


# ---------- 评测：和学生自己走一遍，和老师对比 ----------
def load_student(path):
    """读权重，返回 (net, 可以直接喂给 run_trial(policy=...) 的函数)。"""
    ckpt = torch.load(CODE_DIR / path, map_location="cpu", weights_only=False)
    net = build_net(ckpt["in_dim"], ckpt["out_dim"], ckpt["hidden"])
    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    def policy(obs):
        with torch.no_grad():
            return net(torch.from_numpy(np.asarray(obs, np.float32))[None]).numpy()[0].astype(np.float32)

    return net, policy, ckpt


def evaluate(args):
    r25 = load_r25()
    net, student, ckpt = load_student(args.eval)
    print(f"学生：{ckpt['in_dim']} → {ckpt['hidden']} → {ckpt['out_dim']}，训练到第 {ckpt['epoch']} 轮，"
          f"验证 MSE {ckpt['val_loss']:.5f}")

    rows = []
    for vx in EVAL_CMDS:
        for name, policy in (("老师", None), ("学生", student)):
            res = [r25.run_trial(vx, 0.0, seed=s, policy=policy) for s in EVAL_SEEDS]
            fr = float(r25.fall_rate(res))
            avx = float(r25.mean_speed(res, 2))
            rows.append([vx, name, round(avx, 3) if not np.isnan(avx) else "nan",
                         round(abs(avx - vx), 3) if not np.isnan(avx) else "nan", fr])
            print(f"  指令 {vx:.1f}  {name}  实际 vx {avx:.3f}  摔倒率 {fr:.0%}")
    write_csv(RESULT_DIR / "bc_eval.csv", ["指令vx", "策略", "实际vx", "跟踪误差", "摔倒率"], rows)
    plot_eval(rows)
    print("结果：results/bc_eval.csv, results/bc_eval.png")


def plot_eval(rows):
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, k, title in ((axes[0], 2, "实际速度 vs 指令"), (axes[1], 4, "摔倒率")):
        for name, color in (("老师", "#2a78d6"), ("学生", "#eb6834")):
            sel = [r for r in rows if r[1] == name]
            x = [r[0] for r in sel]
            y = [float(r[k]) * (100 if k == 4 else 1) for r in sel]
            ax.plot(x, y, "-o", color=color, linewidth=2, markersize=7, label=name)
        if k == 2:
            ax.plot([0.25, 0.65], [0.25, 0.65], "--", color="#9a9893", linewidth=1, label="理想跟踪")
            ax.set(ylabel="实际 vx (m/s)")
        else:
            ax.set(ylabel="摔倒率 (%)", ylim=(-5, 105))
        ax.set(xlabel="指令 vx (m/s)", title=title)
        ax.grid(alpha=0.3, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
    fig.suptitle("行为克隆学生 vs 老师（每组 5 个种子，15 s）")
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "bc_eval.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=DATA_FILE, help=f"训练数据（相对 code 目录），默认 {DATA_FILE}")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help=f"训练轮数，默认 {EPOCHS}")
    parser.add_argument("--limit", type=int, default=0, help="只用这么多训练步（小规模试跑用），默认全部")
    parser.add_argument("--eval", default="", help="不训练，直接评测这个权重文件")
    args = parser.parse_args()
    if args.eval:
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()


