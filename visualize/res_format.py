import json
import os
import pandas as pd
from experiments import experiments

OUTPUT_DIR = './output'


def extract_best_ap(folder):
    """从 metrics.json 提取最高 AP 和 AP50 及对应 iteration"""
    path = os.path.join(OUTPUT_DIR, folder, 'metrics.json')
    if not os.path.exists(path):
        return None

    best_ap, best_ap50 = -1, -1
    best_ap_iter, best_ap50_iter = -1, -1

    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            if 'bbox/AP' not in data:
                continue
            it = data.get('iteration', -1)
            ap = data['bbox/AP']
            ap50 = data.get('bbox/AP50', -1)

            if ap > best_ap:
                best_ap = ap
                best_ap_iter = it
            if ap50 > best_ap50:
                best_ap50 = ap50
                best_ap50_iter = it

    if best_ap < 0:
        return None
    return best_ap, best_ap_iter, best_ap50, best_ap50_iter


rows = []
for folder, model, backbone, lr, batch, max_iter, dataset in experiments:
    result = extract_best_ap(folder)
    if result is None:
        print(f"[跳过] {folder}: metrics.json 不存在或无 AP 记录")
        continue
    best_ap, ap_iter, best_ap50, ap50_iter = result
    rows.append({
        'Experiment': folder,
        'Model': model,
        'Backbone': backbone,
        'Dataset': dataset,
        'LR': lr,
        'Batch': batch,
        'MaxIter': max_iter,
        'Best AP': round(best_ap, 2),
        'Best AP50': round(best_ap50, 2),
    })

df = pd.DataFrame(rows)
df = df.sort_values('Best AP50', ascending=False).reset_index(drop=True)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', 40)

print("\n" + "=" * 120)
print("  训练过程中最佳 AP / AP50 汇总")
print("=" * 120)
print(df.to_string(index=False))
print("=" * 120)

# save_path = os.path.join(OUTPUT_DIR, 'best_ap_summary.csv')
# df.to_csv(save_path, index=False)
# print(f"\n已保存到: {save_path}")
