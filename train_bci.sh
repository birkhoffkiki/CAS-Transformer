# use AMP
export CUDA_VISIBLE_DEVICES=6,7
python -m torch.distributed.launch --nproc_per_node 2 --master_port 12345  train_bci.py \
--cfg configs/BCIData/train.yaml --batch-size 2 \
--amp-opt-level O1
