import os
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch, hooks
from detectron2.evaluation import COCOEvaluator, verify_results


class FasterRCNNTrainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    def build_hooks(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(),
        ]

        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(
                self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
            ))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))

        return ret


def register_rsna_dataset():
    data_root = "./dataset/RSNA"

    for split, folder in [("rsna_train", "train2017"), ("rsna_val", "val2017")]:
        json_file = f"{data_root}/annotations/{split.replace('rsna_', '')}_data.json"
        image_root = f"{data_root}/{folder}"

        if split in DatasetCatalog:
            continue

        DatasetCatalog.register(
            split, lambda j=json_file, r=image_root, s=split: load_coco_json(j, r, s)
        )
        MetadataCatalog.get(split).set(
            thing_classes=["pneumonia"],
            evaluator_type="coco",
            json_file=json_file,
            image_root=image_root,
        )


def setup(args):
    cfg = get_cfg()
    register_rsna_dataset()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = DefaultTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = FasterRCNNTrainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = FasterRCNNTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
