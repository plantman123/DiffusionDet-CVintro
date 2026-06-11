CUDA_VISIBLE_DEVICES=4,5,6,7 python train_fasterrcnn.py \
 --num-gpus 4 \
 --config-file ./faster_rcnn.rsna.res50.yaml \
 OUTPUT_DIR ./output/fasterrcnn_res50 \
 SOLVER.CHECKPOINT_PERIOD 1000 \
 TEST.EVAL_PERIOD 1000
