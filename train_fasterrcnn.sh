# CUDA_VISIBLE_DEVICES=4,5,6,7 python train_fasterrcnn.py \
#  --num-gpus 4 \
#  --dist-url tcp://127.0.0.1:50161 \
#  --config-file ./configs/faster_rcnn.rsna.res101.yaml \
#  --dataroot ./dataset/LRSNA \
#  SOLVER.MAX_ITER 6000 \
#  OUTPUT_DIR ./output/lrsna_fasterrcnn_res101 \
#  SOLVER.CHECKPOINT_PERIOD 1000 \
#  TEST.EVAL_PERIOD 1000


CUDA_VISIBLE_DEVICES=0,1,2,3 python train_fasterrcnn.py \
 --num-gpus 4 \
 --dist-url tcp://127.0.0.1:50162 \
 --config-file ./configs/faster_rcnn.rsna.res101.yaml \
 --dataroot ./dataset/RSNA \
 SOLVER.MAX_ITER 5000 \
 OUTPUT_DIR ./output/fasterrcnn_res101 \
 SOLVER.CHECKPOINT_PERIOD 1000 \
 TEST.EVAL_PERIOD 1000
 