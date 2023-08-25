model=semba
dataset=BitcoinOTC-1
task=signlink_class # sign_class, link_class, signlink_class
num_epoch=10
lr_init=0.01
device=cpu

python train.py \
    --model ${model} \
    --dataset ${dataset} \
    --task ${task} \
    --num_epochs ${num_epoch} \
    --lr_init ${lr_init} \
    --num_feats 8 \
    --feat_type zeros \
    --null_nsamples 1 \
    --to_save \
    --device ${device}