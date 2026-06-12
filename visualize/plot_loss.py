import json
import os
import matplotlib.pyplot as plt
import numpy as np

from experiments import experiments

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = './output'


def load_metrics(folder):
    """从 metrics.json 中提取 iteration 和 total_loss"""
    path = os.path.join(OUTPUT_DIR, folder, 'metrics.json')
    iterations, losses = [], []
    if not os.path.exists(path):
        print(f'{folder} 不存在')
        return 0, np.array([]), np.array([])
    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            if 'total_loss' in data and 'iteration' in data:
                iterations.append(data['iteration'])
                losses.append(data['total_loss'])
    return 1, np.array(iterations), np.array(losses)

def smooth(values, weight=0.9):
    """指数移动平均平滑"""
    smoothed = []
    last = values[0]
    for v in values:
        last = weight * last + (1 - weight) * v
        smoothed.append(last)
    return np.array(smoothed)

colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('Training Total Loss ', fontsize=16, fontweight='bold')

for idx, (folder, model, backbone, lr, batch, max_iter, dataset) in enumerate(experiments):
    success, iters, losses = load_metrics(folder)
    if not success:
        continue
    label = f'{model}_{backbone}_lr{lr}_bs{batch} (iter={max_iter}, {dataset})'
    color = colors[idx]

    ax1.plot(iters, losses, alpha=0.15, linewidth=0.6, color=color)
    ax1.plot(iters, smooth(losses, 0.9), linewidth=2, label=label, color=color)

    ax2.plot(iters, smooth(losses, 0.95), linewidth=2, label=label, color=color)

for ax in (ax1, ax2):
    ax.set_xlim(0, 10000)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Total Loss', fontsize=12)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

ax1.set_title('Raw + EMA Smoothed (weight=0.9)', fontsize=13)
ax2.set_title('EMA Smoothed (weight=0.95)', fontsize=13)

plt.tight_layout()
save_path = os.path.join("./visualize", 'loss_all_experiments.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f'已保存: {save_path}')
plt.close()

print('绘制完成！')
