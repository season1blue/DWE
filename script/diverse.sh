#!bin/bash

export DIR_CODE="../nel_model"

export DIR_SEARCH="../../data/wikidiverse/search_top100.json"
export PATH_ANS="../../data/wikidiverse/entity_list.json"
export PATH_NEG_CONFIG="../../data/wikidiverse/neg.json"
export DIR_PREPRO="../../data/wikidiverse"
export DATASET="diverse"
export IMG_PATH="../../data/ImgData"
export GT_TYPE="brief"  #brief

#export OVERWRITE=False

export MODE="train"

export EPOCHS=500  #300
export LOGGING_STEPS=565
export SAVE_STEPS=500 #1000  # 0 represent not save
export BATCH_SIZE=128
export MAX_SENT_LENGTH=32
export DROPOUT=0.4
export DECAY=0.001

export IMG_FEAT_SIZE=512
export TEXT_FEAT_SIZE=512

export LR=5e-6

export NUM_ATTEN_LAYERS=2
export NEG_SAMPLE_NUM=1
export HIDDEN_SIZE=512
export NUM_HEADERS=8
export FF_SIZE=2048
export OUTPUT_SIZE=512

export FEAT_CATE="w"
export LOSS_FUNCTION="triplet"
export LOSS_MARGIN=0.5
export SIMILARITY="cos"
export LOSS_SCALE=16
export GPU=0


export SEED=114514



python $DIR_CODE/train.py --dir_prepro $DIR_PREPRO \
--path_ans_list $PATH_ANS \
--dir_img_feat $DIR_PREPRO \
--dir_neg_feat $DIR_PREPRO \
--logging_steps $LOGGING_STEPS \
--save_steps $SAVE_STEPS \
--per_gpu_train_batch_size $BATCH_SIZE \
--per_gpu_eval_batch_size $BATCH_SIZE \
--neg_sample_num $NEG_SAMPLE_NUM \
--model_type bert \
--num_train_epochs $EPOCHS \
--overwrite_output_dir \
--strip_accents \
--path_candidates $DIR_SEARCH \
--path_neg_config $PATH_NEG_CONFIG \
--num_attn_layers $NUM_ATTEN_LAYERS \
--loss_scale $LOSS_SCALE \
--loss_margin $LOSS_MARGIN \
--loss_function $LOSS_FUNCTION \
--similarity $SIMILARITY \
--feat_cate $FEAT_CATE \
--learning_rate $LR \
--dropout $DROPOUT \
--weight_decay $DECAY \
--hidden_size $HIDDEN_SIZE \
--nheaders $NUM_HEADERS \
--ff_size $FF_SIZE \
--output_size $OUTPUT_SIZE \
--do_train \
--evaluate_during_training \
--gpu_id $GPU \
--seed $SEED \
--max_sent_length $MAX_SENT_LENGTH \
--img_feat_size $IMG_FEAT_SIZE \
--text_feat_size $TEXT_FEAT_SIZE \
--dataset $DATASET \
--img_path $IMG_PATH \
--gt_type $GT_TYPE \
#--overwrite_cache $OVERWRITE