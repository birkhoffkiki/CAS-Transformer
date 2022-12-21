export CUDA_VISIBLE_DEVICES=4,5

python -m torch.distributed.launch --nproc_per_node 2 --master_port 12259  train_aperio.py \
--cfg configs/AperioData/train.yaml --batch-size 2 \
--amp-opt-level O1
