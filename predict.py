import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import torch

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model

CSV_PATH = "dataset/raw-RSNA/stage_2_train_labels.csv"


def setup_cfg_diffdet(args):
    """DiffusionDet 配置"""
    from diffusiondet import add_diffusiondet_config
    from diffusiondet.util.model_ema import add_model_ema_configs

    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def setup_cfg_fasterrcnn(args):
    """Faster R-CNN 配置"""
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def load_model_diffdet(cfg):
    from diffusiondet.util.model_ema import (
        may_get_ema_checkpointer, EMADetectionCheckpointer,
    )

    model = build_model(cfg)
    model.eval()
    kwargs = may_get_ema_checkpointer(cfg, model)
    if cfg.MODEL_EMA.ENABLED:
        EMADetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR, **kwargs).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=False,
        )
    else:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR, **kwargs).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=False,
        )
    return model


def load_model_fasterrcnn(cfg):
    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=False,
    )
    return model


def run_inference(cfg, model, original_image):
    import detectron2.data.transforms as T

    if cfg.INPUT.FORMAT == "RGB":
        image_for_model = original_image[:, :, ::-1]  # BGR → RGB
    else:
        image_for_model = original_image

    height, width = original_image.shape[:2]
    aug = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST],
        cfg.INPUT.MAX_SIZE_TEST,
    )
    image = aug.get_transform(image_for_model).apply_image(image_for_model)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    image = image.to(cfg.MODEL.DEVICE)

    batched_input = {"image": image, "height": height, "width": width}

    with torch.no_grad():
        outputs = model([batched_input])[0]

    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes.tensor  # (N, 4) xyxy
    scores = instances.scores             # (N,)
    return boxes, scores


def load_gt_boxes(csv_path, patient_id):
    """从 CSV 中读取某个 patientId 的所有 GT 框，返回 (N, 4) xyxy 格式的 ndarray"""
    df = pd.read_csv(csv_path)
    rows = df[(df["patientId"] == patient_id) & (df["Target"] == 1)]
    if rows.empty:
        return np.empty((0, 4), dtype=np.float32)

    gt_boxes = []
    for _, row in rows.iterrows():
        x, y, w, h = row["x"], row["y"], row["width"], row["height"]
        gt_boxes.append([x, y, x + w, y + h])
    return np.array(gt_boxes, dtype=np.float32)

def visualize(image_rgb, pred_boxes, pred_scores, gt_boxes,
              score_thresh=0.3, max_pred=50):
    """
    绘制预测框和 GT 框。
    返回 matplotlib Figure。
    """
    H, W = image_rgb.shape[:2]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image_rgb)

    # ── 绘制 GT 框（浅蓝色, 先画使其在底层） ──
    for box in gt_boxes:
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2.5,
            edgecolor=(0.4, 0.75, 1.0),  # 浅蓝色
            facecolor=(0.4, 0.75, 1.0, 0.10),
            linestyle="-",
        )
        ax.add_patch(rect)
        ax.text(x1, y1 - 4, "GT", fontsize=8, color=(0.4, 0.75, 1.0),
                fontweight="bold", va="bottom")

    keep = pred_scores > score_thresh
    pred_boxes = pred_boxes[keep]
    pred_scores = pred_scores[keep]

    if len(pred_scores) > max_pred:
        _, topk = pred_scores.topk(max_pred)
        pred_boxes = pred_boxes[topk]
        pred_scores = pred_scores[topk]

    boxes_np = pred_boxes.numpy()
    scores_np = pred_scores.numpy()
    order = np.argsort(scores_np)
    boxes_np = boxes_np[order]
    scores_np = scores_np[order]

    for box, score in zip(boxes_np, scores_np):
        x1, y1, x2, y2 = box
        w_box, h_box = x2 - x1, y2 - y1
        x1, y1 = max(0, x1), max(0, y1)
        w_box = min(W - x1, w_box)
        h_box = min(H - y1, h_box)
        if w_box <= 0 or h_box <= 0:
            continue
        rect = patches.Rectangle(
            (x1, y1), w_box, h_box,
            linewidth=2.0,
            edgecolor=(1.0, 0.15, 0.15),  # 红色
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(x1, y1 - 4, f"{score:.2f}", fontsize=7,
                color=(1.0, 0.15, 0.15), fontweight="bold", va="bottom")

    legend_handles = [
        patches.Patch(edgecolor=(1.0, 0.15, 0.15), facecolor="none",
                      linewidth=2, label="Prediction"),
        patches.Patch(edgecolor=(0.4, 0.75, 1.0),
                      facecolor=(0.4, 0.75, 1.0, 0.10),
                      linewidth=2, label="Ground Truth"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=10)
    ax.axis("off")

    n_gt = len(gt_boxes)
    n_pred = int(keep.sum()) if isinstance(keep, torch.Tensor) else len(boxes_np)
    ax.set_title(
        f"Pred(red): {n_pred}  |  GT(blue): {n_gt}  |  thresh={score_thresh}",
        fontsize=12,
    )

    plt.tight_layout()
    return fig


def get_parser():
    parser = argparse.ArgumentParser(
        description="加载模型对单张图片进行推理，绘制预测框(红)与GT框(浅蓝)")
    parser.add_argument(
        "--model-type", default="diffdet",
        choices=["diffdet", "fasterrcnn"],
        help="模型类型: diffdet | fasterrcnn",
    )
    parser.add_argument("--config-file", required=True, metavar="FILE",
                        help="Detectron2 配置文件路径")
    parser.add_argument("--input", required=True, nargs="+",
                        help="输入图像路径，支持多张图片")
    parser.add_argument("--output", default="output/vis_pred",
                        help="输出目录")
    parser.add_argument("--csv", default=CSV_PATH,
                        help="GT 标注 CSV 路径")
    parser.add_argument("--score-thresh", type=float, default=0.3,
                        help="预测框置信度阈值")
    parser.add_argument("--dataroot", default=None,
                        help="数据集根目录（仅 fasterrcnn 需要，用于注册数据集）")
    parser.add_argument(
        "--opts", default=[], nargs=argparse.REMAINDER,
        help="覆盖配置项, 例如: MODEL.WEIGHTS path/to/model.pth",
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    if args.model_type == "diffdet":
        cfg = setup_cfg_diffdet(args)
        model = load_model_diffdet(cfg)
    else:
        if args.dataroot is None:
            args.dataroot = "dataset/RSNA"
        from detectron2.data import DatasetCatalog, MetadataCatalog
        from detectron2.data.datasets import load_coco_json
        for split, folder in [("rsna_train", "train2017"), ("rsna_val", "val2017")]:
            json_file = f"{args.dataroot}/annotations/{split.replace('rsna_', '')}_data.json"
            image_root = f"{args.dataroot}/{folder}"
            if split not in DatasetCatalog:
                DatasetCatalog.register(
                    split, lambda j=json_file, r=image_root, s=split: load_coco_json(j, r, s)
                )
                MetadataCatalog.get(split).set(
                    thing_classes=["pneumonia"], evaluator_type="coco",
                    json_file=json_file, image_root=image_root,
                )
        cfg = setup_cfg_fasterrcnn(args)
        model = load_model_fasterrcnn(cfg)

    model.to(cfg.MODEL.DEVICE)
    logger.info("模型加载完成")

    os.makedirs(args.output, exist_ok=True)
    gt_df = pd.read_csv(args.csv)

    for img_path in args.input:
        original_image = read_image(img_path, format="BGR")
        image_rgb = original_image[:, :, ::-1].copy()

        pred_boxes, pred_scores = run_inference(cfg, model, original_image)

        patient_id = os.path.splitext(os.path.basename(img_path))[0]
        rows = gt_df[(gt_df["patientId"] == patient_id) & (gt_df["Target"] == 1)]
        if rows.empty:
            gt_boxes = np.empty((0, 4), dtype=np.float32)
        else:
            gt_boxes = np.array(
                [[r["x"], r["y"], r["x"] + r["width"], r["y"] + r["height"]]
                 for _, r in rows.iterrows()],
                dtype=np.float32,
            )

        logger.info(f"[{patient_id}] 预测框: {len(pred_scores)}, GT框: {len(gt_boxes)}")

        fig = visualize(image_rgb, pred_boxes, pred_scores, gt_boxes,
                        score_thresh=args.score_thresh)

        out_path = os.path.join(args.output, f"{patient_id}_pred_vs_gt.png")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  -> {out_path}")
