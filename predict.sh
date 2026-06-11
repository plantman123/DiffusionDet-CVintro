# python predict_and_visualize.py \
#     --config-file configs/diffdet.rsna.res50.yaml \
#     --input dataset/RSNA/train2017/0a8ccb49-debc-4e9a-b5dc-eefc3fe909ca.jpg \
#     --output output/vis_pred \
#     --score-thresh 0.3 \
#     --opts MODEL.WEIGHTS output/MAXITER_5000_BATCH_64_res50/model_final.pth

python predict.py \
    --model-type fasterrcnn \
    --config-file configs/faster_rcnn.rsna.res50.yaml \
    --input dataset/RSNA/train2017/fd6bfade-0ea6-4048-a301-a688fafda191.jpg \
    --output eval/predict \
    --score-thresh 0.3 \
    --opts MODEL.WEIGHTS output/fasterrcnn_res50/model_final.pth