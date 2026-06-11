import json
import os
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = './output'

experiments = [
    ('MAXITER_2500_BATCH_32_res101', 'DiffusionDet', 'Res101', 8e-5,  32,  2500, 'sRSNA'),
    ('MAXITER_2500_BATCH_48_res50',  'DiffusionDet', 'Res50',  5e-5,  32,  2500, 'sRSNA'),
    ('MAXITER_5000_BATCH_64_res50',  'DiffusionDet', 'Res50',  8e-5,  56,  5000, 'sRSNA'),
    ('fasterrcnn_res50',             'FasterRCNN',   'Res50',  5e-3,  16,  5000, 'sRSNA'),
    ('fasterrcnn_res101',            'FasterRCNN',   'Res101', 5e-3,  16,  5000, 'sRSNA')
    ('optres50',                     'DiffusionDet', 'Res50',  8e-5,  56,  8000, 'sRSNA'),
    ('optres101',                    'DiffusionDet', 'Res101', 8e-5,  56,  8000, 'sRSNA'),
    ('MAXITER_10000_BATCH_64_res50', 'DiffusionDet', 'Res50',  1.5e-4,64,  10000,'sRSNA'),
    ('MAXITER_10000_BATCH_64_res101','DiffusionDet', 'Res101', 5e-5,  64,  10000,'sRSNA'),
    ('lrsna_fasterrcnn_res50',       'FasterRCNN',   'Res50',  5e-3,  16,  6000, 'lRSNA'),
    ('lrsna_fasterrcnn_res101',      'FasterRCNN',   'Res101', 5e-3,  16,  6000, 'lRSNA'),
    ('lrsna_optres50',               'DiffusionDet', 'Res50',  8e-5,  32,  3000, 'lRSNA'),
    ('lrsna_optres101',              'DiffusionDet', 'Res101', 8e-5,  32,  3000, 'lRSNA'),
]

def load_metrics(folder):
    """从 metrics.json 中提取 iteration 和 total_loss"""
    path = os.path.join(OUTPUT_DIR, folder, 'metrics.json')
    iterations, losses = [], []
    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            if 'total_loss' in data and 'iteration' in data:
                iterations.append(data['iteration'])
                losses.append(data['total_loss'])
    return np.array(iterations), np.array(losses)

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
fig.suptitle('Training Total Loss (All Experiments)', fontsize=16, fontweight='bold')

for idx, (folder, model, backbone, lr, batch, max_iter) in enumerate(experiments):
    iters, losses = load_metrics(folder)
    label = f'{model}_{backbone}_lr{lr}_bs{batch} (iter={max_iter})'
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
save_path = os.path.join(OUTPUT_DIR, 'loss_all_experiments.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f'已保存: {save_path}')
plt.close()

print('绘制完成！')
