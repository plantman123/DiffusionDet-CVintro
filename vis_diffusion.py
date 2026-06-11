import argparse
import glob
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from diffusiondet import add_diffusiondet_config
from diffusiondet.util.model_ema import add_model_ema_configs, may_get_ema_checkpointer, EMADetectionCheckpointer

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model


def setup_cfg(args):
    """加载配置"""
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.freeze()
    return cfg


def load_model(cfg):
    """加载模型"""
    model = build_model(cfg)
    model.eval()

    kwargs = may_get_ema_checkpointer(cfg, model)
    if cfg.MODEL_EMA.ENABLED:
        EMADetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR, **kwargs).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=False
        )
    else:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR, **kwargs).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=False
        )
    return model


def preprocess_image(cfg, original_image):
    """图像预处理"""
    import detectron2.data.transforms as T
    if cfg.INPUT.FORMAT == "RGB":
        original_image = original_image[:, :, ::-1]  # BGR -> RGB

    height, width = original_image.shape[:2]

    # Resize
    aug = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST],
        cfg.INPUT.MAX_SIZE_TEST,
    )
    image = aug.get_transform(original_image).apply_image(original_image)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    image = image.to(cfg.MODEL.DEVICE)

    return {"image": image, "height": height, "width": width}


def draw_boxes_on_ax(ax, image_np, boxes, scores, H, W, score_thresh=0.05, max_boxes=50):
    """在绘制检测框"""
    # 过滤
    keep = scores > score_thresh
    boxes = boxes[keep]
    scores = scores[keep]

    if len(scores) > max_boxes:
        _, topk = scores.topk(min(max_boxes, len(scores)))
        boxes = boxes[topk]
        scores = scores[topk]

    boxes_np = boxes.numpy()
    scores_np = scores.numpy()
    order = np.argsort(scores_np)
    boxes_np = boxes_np[order]
    scores_np = scores_np[order]

    for box, score in zip(boxes_np, scores_np):
        x1, y1, x2, y2 = box
        w_box, h_box = x2 - x1, y2 - y1
        x1, y1 = max(0, x1), max(0, y1)
        w_box, h_box = min(W - x1, w_box), min(H - y1, h_box)
        if w_box <= 0 or h_box <= 0:
            continue
        intensity = min(1.0, float(score))
        color = (1.0, 0.15 * (1 - intensity), 0.15 * (1 - intensity))
        rect = patches.Rectangle(
            (x1, y1), w_box, h_box, linewidth=1.2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)


def visualize_progressive_denoising(image_np, steps_data, class_names, score_thresh=0.05):
    """
    渐进去噪的可视化大图；
    上行: 预测的干净框(x_start), 即模型当前对最终检测结果的估计
    下行: 当前带噪声的输入框(noisy boxes), 即作为模型输入的噪声框

    Args:
        image_np: 原始图像 (H, W, 3) RGB uint8
        steps_data: ddim_sample_visualization 返回的中间步骤列表
        class_names: 类别名列表
        score_thresh: 框的置信度阈值
    """
    n_steps = len(steps_data)
    fig, axes = plt.subplots(2, n_steps, figsize=(4 * n_steps, 8.5))
    if n_steps == 1:
        axes = axes.reshape(2, 1)

    H, W = image_np.shape[:2]

    for idx, step in enumerate(steps_data):
        time_val = step['time']
        time_next = step['time_next']
        boxes = step['boxes']
        scores = step['scores']
        noisy_boxes = step['noisy_boxes']

        # 预测结果
        ax_pred = axes[0, idx]
        ax_pred.imshow(image_np)
        draw_boxes_on_ax(ax_pred, image_np, boxes, scores, H, W, score_thresh, max_boxes=30)
        if time_next < 0:
            ax_pred.set_title(f"Step {idx+1}: T={time_val} → Final\n(Predicted boxes)", fontsize=9)
        else:
            ax_pred.set_title(f"Step {idx+1}: T={time_val} → T={time_next}\n(Predicted boxes)", fontsize=9)
        ax_pred.axis('off')

        # 初始噪声
        ax_noisy = axes[1, idx]
        ax_noisy.imshow(image_np)
        # 噪声框没有分数，统一用浅灰色
        noisy_np = noisy_boxes.numpy()
        for box in noisy_np[:40]:
            x1, y1, x2, y2 = box
            w_box, h_box = x2 - x1, y2 - y1
            x1, y1 = max(0, x1), max(0, y1)
            w_box, h_box = min(W - x1, w_box), min(H - y1, h_box)
            if w_box <= 0 or h_box <= 0:
                continue
            rect = patches.Rectangle(
                (x1, y1), w_box, h_box,
                linewidth=0.8,
                edgecolor=(0.4, 0.5, 0.7),
                facecolor=(0.5, 0.6, 0.8, 0.15),
                linestyle='--'
            )
            ax_noisy.add_patch(rect)
        ax_noisy.set_title(f"(Noisy input boxes)", fontsize=9)
        ax_noisy.axis('off')

    fig.suptitle("DiffusionDet Progressive Denoising", fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    return fig


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        default="configs/diffdet.rsna.res50.yaml",
        metavar="FILE",
        help="配置文件路径",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="输入图像路径(支持 glob 通配符)",
    )
    parser.add_argument(
        "--output",
        help="输出目录. 若未指定，将用 matplotlib 显示.",
    )
    parser.add_argument(
        "--opts",
        help="覆盖配置项, 例如: MODEL.DiffusionDet.SAMPLE_STEP 5",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    sample_steps = cfg.MODEL.DiffusionDet.SAMPLE_STEP

    logger.info("Loading models...")
    model = load_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    # 获取类别名
    from detectron2.data import MetadataCatalog
    class_names = ["bg"]
    if len(cfg.DATASETS.TEST):
        metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        class_names = metadata.get("thing_classes", ["pneumonia"])

    # 处理输入
    if len(args.input) == 1:
        args.input = glob.glob(os.path.expanduser(args.input[0]))

    for img_path in tqdm.tqdm(args.input, desc="Processing"):
        # 读取图像
        original_image = read_image(img_path, format="BGR")
        image_rgb = original_image[:, :, ::-1]

        # 预处理
        batched_input = preprocess_image(cfg, original_image)

        # forward 推理
        with torch.no_grad():
            images, images_whwh = model.preprocess_image([batched_input])
            src = model.backbone(images.tensor)
            features = [src[f] for f in model.in_features]

            steps_data = model.ddim_sample_visualization(
                [batched_input], features, images_whwh, images
            )
        logger.info(f" Total: {len(steps_data)} steps.")

        # 创建可视化
        fig = visualize_progressive_denoising(
            image_rgb, steps_data, class_names, score_thresh=0.05
        )

        # 保存
        if args.output:
            os.makedirs(args.output, exist_ok=True)
            out_filename = os.path.join(args.output, os.path.basename(img_path))
            out_filename = os.path.splitext(out_filename)[0] + "_denoising.png"
            fig.savefig(out_filename, dpi=450, bbox_inches='tight')
            logger.info(f"  保存到: {out_filename}")
        else:
            matplotlib.use("TkAgg")
            plt.show()

        plt.close(fig)