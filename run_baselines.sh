datasets=(BitcoinOTC-1 BitcoinAlpha-1 wikirfa epinions) # 
models=(gcn sgcn sigat sgclstm tgn semba semba-noprop prop) #
tasks=(signlink_class sign_class link_pred signwt_pred) 
num_epochs=(5 10 15 20) #
lr_inits=(0.01 0.001 0.0001)

for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        for task in "${tasks[@]}"; do
            for num_epoch in "${num_epochs[@]}"; do
                for lr_init in "${lr_inits[@]}"; do
                    if [ "${dataset}" -eq "epinions" ]; then 
                        python train.py \
                            --model ${model} \
                            --dataset ${dataset} \
                            --task ${task} \
                            --num_epochs ${num_epoch} \
                            --lr_init ${lr_init} \
                            --num_feats 8 \
                            --to_save \
                            --feat_type zeros \
                            --null_nsamples 1 \
                            --batch_size 16000 \
                            --device $1;
                    else
                        python train.py \
                            --model ${model} \
                            --dataset ${dataset} \
                            --task ${task} \
                            --num_epochs ${num_epoch} \
                            --lr_init ${lr_init} \
                            --num_feats 8 \
                            --to_save \
                            --feat_type zeros \
                            --null_nsamples 1 \
                            --batch_size 1000 \
                            --device $1;
                    fi
                done
            done
        done
    done
done