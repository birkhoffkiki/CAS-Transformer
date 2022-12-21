export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8

python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345 pretrain.py \
--cfg configs/pretrain.yaml --batch-size 1