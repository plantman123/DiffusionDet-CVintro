python vis_diffusion.py \
  --config-file configs/diffdet.rsna.res50.yaml \
  --input < your input img > \
  --output < your output folder > \
  --opts MODEL.WEIGHTS < your checkpoint path > \
  MODEL.DiffusionDet.SAMPLE_STEP < sample steps, suggest 4-6 >